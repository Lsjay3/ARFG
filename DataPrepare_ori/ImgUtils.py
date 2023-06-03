# Jay的开发时间：2022/10/4  15:46
# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re

import PIL.Image as Image


# 定义图像拼接函数
def compose_images(image_path, image_save_path, row, col):
    IMAGES_PATH = image_path  # 图片集来源地址
    IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
    IMAGE_SIZE = 15 * 15  # 每张小图片的大小
    IMAGE_SAVE_PATH = image_save_path  # 图片转换后的地址
    # 获取图片集地址下的所有图片名称
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]
    image_names.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    # print(image_names)
    # l = len(image_names)
    IMAGE_ROW = row  # 图片间隔，也就是合并成一张图后，一共有几行
    IMAGE_COLUMN = col  # 图片间隔，也就是合并成一张图后，一共有几列
    # # 简单的对于参数的设定和实际图片集的大小进行数量判断
    # if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    #     raise ValueError("合成图片的参数和要求的数量不能匹配！")

    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + '\\' + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            # from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    # to_image.resize(32, 32)
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

# del_file(r'E:\笨比j\rfid\Impinj R420\img\\')
# image_compose()  # 调用函数这里插入代码片
