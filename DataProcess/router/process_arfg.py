# Jay的开发时间：2023/6/3  12:07
import numpy
from config.config import Config
from router.process_base import ProcessBase
from utils.signal_prepare import together_tags
from utils.signal_process import pant_img


class ProcessARFG(ProcessBase):
    def __init__(self, gesture):
        super().__init__(gesture)

    def together_tags(self, dic_data: dict, array: str):
        together_tags(dic_data=dic_data, array=Config.TAG_ARRAY)

    def pant_img(self, like_hood: numpy.ndarray,
                 start: int, save_tuple: tuple):
        pant_img(like_hood=like_hood, start=start, save_tuple=save_tuple)
