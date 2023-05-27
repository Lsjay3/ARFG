# Jay的开发时间：2022/10/4  15:46
# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re

import imageio
from PIL import Image, ImageOps


def image_compose(image_path: str, image_save_path: str, row: int, col: int):
    """
    将各个时刻的特征图拼接成一张大图，便于观察
    :param image_path: 待拼接图片路径
    :param image_save_path: 图片保存路径
    :param row: 行数
    :param col: 列数
    :return: None
    """
    # 图片集来源地址
    IMAGES_PATH = image_path
    # 图片格式
    IMAGES_FORMAT = ['.jpg', '.JPG']
    # 每张小图片的大小
    IMAGE_SIZE = 15 * 15
    # 图片转换后的地址
    IMAGE_SAVE_PATH = image_save_path
    # 获取图片集地址下的所有图片名称
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]
    image_names.sort(key=lambda l: int(re.findall(r'\d+', l)[0]))

    # 图片间隔，也就是合并成一张图后，一共有几行
    IMAGE_ROW = row
    # 图片间隔，也就是合并成一张图后，一共有几列
    IMAGE_COLUMN = col

    # 创建一个新图
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)


def png2gif(file_list: list, name: str, duration: float = 0.5):
    """
    将多张图片组合成gif
    :param file_list: 图片位置列表
    :param name: 处理之后的gif名
    :param duration: gif持续时长
    :return: None
    """
    frames = []
    for img in file_list:
        crop_margin(img, img)
        frames.append(imageio.imread(img))
    imageio.mimsave(name, frames, 'GIF', duration=duration)


def crop_margin(img_file_obj, out, padding=(0, 0, 0, 0)):
    """
    用于裁剪出图像中的主要内容区域，同时在主要内容周围添加一些边缘空白
    :param img_file_obj: 指向要打开的图像文件的路径或文件对象
    :param out: 一个字符串，表示保存裁剪后图像的文件路径
    :param padding: 一个包含四个整数的列表或元组，表示在四个方向上分别添加的空白大小。
    :return: None
    """
    image = Image.open(img_file_obj).convert('RGB')
    ivt_image = ImageOps.invert(image)
    bbox = ivt_image.getbbox()
    left = bbox[0] - padding[0]
    top = bbox[1] - padding[1]
    right = bbox[2] + padding[2]
    bottom = bbox[3] + padding[3]
    cropped_image = image.crop([left, top, right, bottom])
    cropped_image.save(out)
