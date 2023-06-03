# Jay的开发时间：2022/10/30  11:09
import os
import re

import imageio
from PIL import Image, ImageOps


def png2gif(filelist, name, duration=0.5):
    frames = []
    for img in filelist:
        crop_margin(img, img)
        frames.append(imageio.imread(img))
    imageio.mimsave(name, frames, 'GIF', duration=0.2)


def crop_margin(img_fileobj, out, padding=(0, 0, 0, 0)):
    image = Image.open(img_fileobj).convert('RGB')
    ivt_image = ImageOps.invert(image)
    bbox = ivt_image.getbbox()
    left = bbox[0] - padding[0]
    top = bbox[1] - padding[1]
    right = bbox[2] + padding[2]
    bottom = bbox[3] + padding[3]
    cropped_image = image.crop([left, top, right, bottom])
    cropped_image.save(out)
    return


path = r'E:\笨比j\rfid\Impinj R420\img'
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
image_names = [name for name in os.listdir(path) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names.sort(key=lambda l: int(re.findall('\d+', l)[0]))
Toplist = []
for i in range(len(image_names)):
    Toplist.append(os.path.join(path, image_names[i]))
# Toplist = glob.glob(os.path.join(path, "*.jpg" ))
# Toplist.sort(key=lambda l: int(re.findall('\d+', l)[0]))
png2gif(Toplist, "move.gif")
