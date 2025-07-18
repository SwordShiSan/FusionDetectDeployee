
from models.experimental import attempt_load
from utils.general import non_max_suppression_obb, scale_coords, letterbox, scale_polys
from utils.torch_utils import select_device

from random import randint
from utils.plots import Annotator, colors
from utils.rboxs_utils import poly2rbox, rbox2poly

import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from models.myfusionnet import FusionNet
from utils.image_transform import *
from utils.evaluator import calculate_metrics

class Detector(object):

    def __init__(self):
        self.img_size = 512
        self.threshold = 0.4
        self.max_frame = 160
        self.init_model()

    def init_model(self):

        self.weights = 'weights/best.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        # !!!slow_conv2d_cpu不支持半精度,会报错
        # model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
        self.colors = [
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names
        ]

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # !!!slow_conv2d_cpu不支持半精度,会报错
        # img = img.half()  # 半精度
        # img /= 255.0  # 图像归一化
        img=img.div(255.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color,
                          thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{:.2f}'.format(cls_id, conf), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return image

    def detect(self, ims):
        
        im0_vi, img_vi = self.preprocess(ims[0])
        im0_ir, img_ir = self.preprocess(ims[1])
        img = torch.cat([img_vi, img_ir], dim=1)
        pred, img_fused = self.m(img, augment=False)
        img_fused *= 255.0
        img_fused = np.ascontiguousarray(img_fused.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8'))
        pred = pred[0].float()
        pred = non_max_suppression_obb(pred, 0.5, 0.3, multi_label=True, max_det=1000)

        image_info = {}
        count = 0
        for i, det in enumerate(pred):  # per image
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])

            annotator = Annotator(img_fused, line_width=2, example=str(self.names))
            if len(det):
                pred_poly = scale_polys(img.shape[2:], pred_poly, img_fused.shape)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Write results
                for *poly, conf, cls in reversed(det):
                    x1, y1, x2, y2, x3, y3, x4, y4 = int(poly[0]), int(poly[1]), int(poly[2]), int(poly[3]), int(poly[4]), int(poly[5]), int(poly[6]),  int(poly[7])  
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.poly_label(poly, label, color=colors(c, True))
                    lbl = self.names[c]
                    count += 1
                    key = '{}-{:02}'.format(lbl, count)
                    image_info[key] = ['{}×{}'.format(
                        max(x1, x2, x3, x4) - min(x1, x2, x3, x4), max(y1, y2, y3, y4) - min(y1, y2, y3, y4) ), np.round(float(conf), 3)]
        
            im0 = annotator.result()
            return im0, image_info
                
class FusionDetectot(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_fusion_model("weights/fusion_model.pth")
        self.init_detect_model("weights/yolov8s_best.pt")

    def init_fusion_model(self,model_path):
        self.fusionmodel = FusionNet()
        self.fusionmodel.load_state_dict(torch.load(model_path,map_location='cpu'))
        self.fusionmodel = self.fusionmodel.to(self.device)
        self.fusionmodel.eval()

    def init_detect_model(self,model_path):
        self.detectmodel = YOLO(model_path)

    def predict(self,vi_path):
        print("predict start--------------------------------------")
        self.fusion(vi_path)
        predict_image_path,front_result=self.detect(vi_path)
        quality_image_info=self.metrics(vi_path)
        print("predict over---------------------------------------")
        return predict_image_path,front_result,quality_image_info

    def fusion(self,vi_path):
        # 将融合图像保存到tmp/fusion文件夹下面,去除了_vi,_ir后缀
        ir_path=vi_path.replace("vi","ir")
        vi_img_tensor,ir_img_tensor=self.read_fusion_input(vi_path,ir_path)
        with torch.no_grad():
            vi_Y, vi_Cb, vi_Cr = RGB2YCbCr(vi_img_tensor)
            vi_Y = vi_Y.to(self.device)
            vi_Cb = vi_Cb.to(self.device)
            vi_Cr = vi_Cr.to(self.device)
            # print(vi_Y.shape)
            # print(ir_img_tensor.shape)
            fused_img = self.fusionmodel(vi_Y, ir_img_tensor)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            vi_file_name=os.path.basename(vi_path)
            save_file_name=vi_file_name.split("_")[0]+"."+vi_file_name.split(".")[1]
            save_img(save_file_name,fused_img[0,::])

    def detect(self,vi_path):
        # 对融合图像进行检测,并返回前端数据,字典数据
        vi_file_name=os.path.basename(vi_path)
        save_file_name=vi_file_name.split("_")[0]+"."+vi_file_name.split(".")[1]
        image_path=os.path.join("./tmp/fusion",save_file_name)

        h, w, _ = cv2.imread(image_path).shape
        # print(h, w)
        self.detectmodel.predict(image_path, save=True,imgsz=640, conf=0.5, save_txt=True, save_conf=True, device=self.device)
        image_name = os.path.basename(image_path)
        # print(image_name)
        predict_results_root = "./runs/detect"

        file_dirs = os.listdir(predict_results_root)
        file_dirs = [file_dir for file_dir in file_dirs if file_dir.startswith("predict")]
        file_dirs.sort()
        # print(file_dirs)
        # print(file_dirs[-1])
        predict_image_path = os.path.join(predict_results_root, file_dirs[-1], image_name)
        predict_labels_path = os.path.join(predict_results_root, file_dirs[-1], "labels", image_name.split(".")[0] + ".txt")
        # print(predict_labels_path)
        front_results_dict = self.read_label(predict_labels_path, h, w)
        # print(front_result)
        print("")
        print("detect image save in "+predict_image_path)
        return predict_image_path,front_results_dict
    
    def metrics(self,vi_path):
        # 计算融合指标，返回前端数据,列表数据
        ir_path=vi_path.replace("vi","ir")
        vi_file_name=os.path.basename(vi_path)
        save_file_name=vi_file_name.split("_")[0]+"."+vi_file_name.split(".")[1]
        fusion_path=os.path.join("./tmp/fusion",save_file_name)
        metrics_list=calculate_metrics(vi_path,ir_path,fusion_path)
        print("calculate metrics over")
        return metrics_list

    
    def read_fusion_input(self,vi_path,ir_path):
        image_vi = cv2.cvtColor(cv2.imread(vi_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image_vi = image_vi.astype(np.float32).transpose(2, 0, 1) / 255.0
        image_vi = np.expand_dims(image_vi, axis=0)

        image_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        image_ir = image_ir.astype(np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)
        image_ir = np.expand_dims(image_ir, axis=0)

        return torch.tensor(image_vi),torch.tensor(image_ir)
    
    def read_label(self,label_path, h, w):
        cls_list = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
        cls_num_list = {"People": 1, "Car": 1, "Bus": 1, "Lamp": 1, "Motorcycle": 1, "Truck": 1}
        targets = np.loadtxt(str(label_path), dtype=np.float32)
        # [n*[cls, x, y, w, h]] n是gt框个数
        # print(targets.shape)
        front_results_dict = {}
        for target in targets:
            obj_cls = int(target[0])
            # print(obj_cls)
            obj_name = cls_list[obj_cls] + "-" + str(cls_num_list[cls_list[obj_cls]])
            obj_info = [str(int(w * target[3])) + "×" + str(int(h * target[4])), round(float(target[5]),3)]
            front_results_dict[obj_name] = obj_info
            cls_num_list[cls_list[obj_cls]] = cls_num_list[cls_list[obj_cls]] + 1
        return front_results_dict
