import os
import re

from utils.img_process import png2gif

path = r'E:\笨比j\rfid\Impinj R420\img'
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
image_names = [name for name in os.listdir(path) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names.sort(key=lambda l: int(re.findall(r'\d+', l)[0]))
Top_list = []
for i in range(len(image_names)):
    Top_list.append(os.path.join(path, image_names[i]))
png2gif(Top_list, "move.gif")
