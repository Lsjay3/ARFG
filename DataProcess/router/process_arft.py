# Jay的开发时间：2023/6/3  12:15
from config.config import Config
from router.process_base import ProcessBase
from utils.signal_process import data_to_csv


class ProcessARFT(ProcessBase):
    def __init__(self, gesture):
        super().__init__(gesture)

    def data_to_csv(self, data: list, target: str):
        data_to_csv(data=data, target=Config.GESTURE_DICT[self.gesture])
