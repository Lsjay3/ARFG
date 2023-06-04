import os

from config.config import Config, Path

from router.process_task import ProcessARFG, ProcessARFT

if __name__ == '__main__':

    # 当前目录下面的所有文件
    path_origin = os.listdir(Path.PATH_ORIGIN_DATA)
    path_origin = [ges for ges in path_origin if ges in Config.GESTURE_DICT.keys()]
    model = "arfg"

    for gesture in path_origin:
        if model == "arfg":
            task = ProcessARFG(gesture=gesture)
        elif model == "arft":
            task = ProcessARFT(gesture=gesture)
        else:
            raise ValueError("无效的任务类型")
        task.process()

    if model == "arft":
        task = ProcessARFT(gesture='a')
        task.train_test_split()






