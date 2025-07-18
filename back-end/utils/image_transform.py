import torch
import numpy as np
import cv2

def RGB2YCbCr(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 0.5
    Cr = (R - Y) * 0.713 + 0.5

    Y = Y.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    Cr = Cr.clamp(0.0, 1.0).detach()

    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
                       ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out


def save_img(filename,img, is_norm=True):
    img = img.cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    if is_norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = img.astype(np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite("./tmp/fusion/"+filename,img)
    print("fusion image save in ./tmp/fusion/"+filename)
