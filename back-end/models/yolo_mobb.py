# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
from logging import Logger
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device, time_sync)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def See_P345(a, layer_name):
    images_per_row = 8
    layer_activation = a.cpu().detach().numpy()
    n_feature = layer_activation.shape[1]  # 每层输出的特征层数
    size1 = layer_activation.shape[-2]  # 每层的特征大小
    size2 = layer_activation.shape[-1]  # 每层的特征大小
    print("-->" + layer_name)
    print(layer_activation.shape, size1, size2, n_feature)
    n_cols = n_feature // images_per_row  # 特征图平铺的行数
    display_grid = np.zeros((size1 * n_cols, images_per_row * size2))  # 每层图片大小
    for col in range(n_cols):  # 行扫描
        for row in range(images_per_row):  # 平铺每行
            # print(layer_activation.shape)
            # print(col*images_per_row+row)
            channel_image = layer_activation[0, col * images_per_row + row, :, :]  # 写入 col * images_per_row + row 特征层
            channel_image = channel_image -  channel_image.mean()  # 标准化处理，增加可视化效果
            channel_image = channel_image / channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # print(channel_image.shape)
            # print(display_grid[col*size1:(col+1)*size1, row*size2:(row+1)*size2].shape)
            display_grid[col * size1:(col + 1) * size1, row * size2:(row + 1) * size2] = channel_image  # 写入大图中
    scale = 1. / max(size1, size2)  # 每组图缩放系数
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    # plt.title(layer_name)
    plt.grid(False)
    print(display_grid.shape)
    plt.imsave(layer_name + '.png', display_grid)
    display_grid, layer_activation,  = '', ''

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
        Args:
            x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)

        Return：
            if train:
                x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            else:
                inference (tensor): (b, n_all_anchors, self.no)
                x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy TODO
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # print("1, ch, s, s", 1, ch, s, s)
            m.inplace = self.inplace
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.tensor([8., 16., 32.])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # 一般 model(x, x2, augment) 调用的都是这个函数并执行 _forward_once
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile=profile, visualize=visualize)  # single-scale inference, train 

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        fusion_out = []
        save = True
        for idx, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                self._profile_one_layer(m, x, dt)

            x = m(x) # run
            y.append(x if m.i in self.save else None)  # save output
            if idx == 0: # 将 fusion 的结果单独保存下来
                fusion_out.append(x)
            if visualize and save:
                feature_visualization(x, m.type, m.i, n=9) # 原本带的 
                # if idx == 4:
                #     See_P345(x, 'runs/vis/val_batch' + str(idx) +'_P3')
                # if idx == 6:
                #     See_P345(x, 'runs/vis/val_batch' + str(idx) +'_P4')
                # if idx == 8:
                #     save = False
                #     See_P345(x, 'runs/vis/val_batch' + str(idx) +'_P5')
        return x, fusion_out[0]

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 185)  # number of outputs = anchors * (classes + 5)      llvip:no=18

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # ch 每一层的 out channel
    # from (当前层输入来自哪些层), number(当前层次数 初定), module(当前层类别), args(当前层类参数 初定)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args

        # eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        
        # ------------------- 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
        # depth gain 控制深度  如 v5s: n*0.33   n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
                 C3CA):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x,
                     C3CA]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        # 增加的一些 Block
        elif m is NIN:
            c2 = args[0]
            args = [c2]
        elif m is SEAttention:
            c2 = args[0]
            args = [c2]
        elif m is OnlyV:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[1]]
            args = [c2]
        elif m is OnlyT:
            c2 = ch[f[1]]
            args = [c2]
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            # print("Add2 arg", args[0])
            args = [c2, args[1]]
        elif m is Concat_bifpn:
            c2 = max([ch[x] for x in f]) #2021.11.11更改配合bifpn使用
        else:
            c2 = ch[f]

        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 打印当层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params  计算这一层的参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 把所有层结构中from不是-1的值记下  [6, 4, 14, 10, 17, 20, 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_) # 将当前层结构module加入layers中
        if i == 0:
            ch = []
        ch.append(c2) # 把当前层的输出channel数加入ch
    return nn.Sequential(*layers), sorted(save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/data2/zcl/code/yolov5_dual_obb/models/transformers/yolov5l-cbamfuse.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='6', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    input_rgb = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    input_ir  = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    input = torch.cat((input_rgb, input_ir), dim=1)

    model = Model(opt.cfg).to(device)
    model.train()
    
    output = model(input)
    if opt.profile:
        img = torch.rand(1 if torch.cuda.is_available() else 1, 6, 640, 640).to(device)
        y = model(img, profile=True)
    print(len(output))
    print(output[0][0].shape)
    print(output[0][1].shape)
    print(output[0][2].shape)
    print(output[1].shape)
    # print(output[2].shape)

    # # Options
    # if opt.line_profile:  # profile layer by layer
    #     _ = model(im, profile=True)

    # elif opt.profile:  # profile forward-backward
    #     results = profile(input=im, ops=[model], n=3)

    # elif opt.test:  # test all models
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')

    # else:  # report fused model summary
    #     model.fuse()
