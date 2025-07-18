import cv2

def predict(dataset, model, ext):
    global img_y
    x_vi = dataset[0].replace('\\', '/') # './tmp/ct/1000_ir.jpg'
    x_ir = x_vi.replace('vi', 'ir')
    file_name = dataset[1]
    print(x_vi)
    print(file_name)
    x_vi = cv2.imread(x_vi)
    x_ir = cv2.imread(x_ir)
    imgs = [x_vi, x_ir]
    img_y, image_info = model.detect(imgs)
    cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
    return image_info