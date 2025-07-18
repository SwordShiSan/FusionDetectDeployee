from core import process, predict


def c_main(path, model, ext):
    image_data = process.pre_process(path)
    image_info = predict.predict(image_data, model, ext)

    return image_data[1] + '.' + ext, image_info

def c_main2(path, model, ext):
    image_data = process.pre_process(path)

    return image_data[1] + '.' + ext

def fusion_and_detect(path,model):
    '''
    传入：可见光图像路径,融合模型和检测模型整合,图片后缀.png或其他
    处理：对红外和可见光图像进行融合,并使用yolov8进行推理出结果
    返回：两个值,一个是检测图像名称(已保存),一个是image_info检测字典
    '''
    predict_image_path,predict_image_info,quality_image_info=model.predict(path)
    return predict_image_path,predict_image_info,quality_image_info


if __name__ == '__main__':
    pass
