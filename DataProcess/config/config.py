class Config:

    # 标签阵列
    TAG_ARRAY = \
        ['E20000191107015312207CD8', 'E20000191107015312307CE0', 'E20000191107015314407D24', 'E20000191107015315907D70',
         'E20000191107015318607DD8',
         'E20000191107015311607CB4', 'E20000191107015313607D04', 'E20000191107015316407D74', 'E20000191107015317907DC0',
         'E20000191107015318007DB4',
         'E20000191107015320507E1C', 'E20000191107015322507E6C', 'E20000191107015322607E78', 'E20000191107015324807EC4',
         'E20000191107015326307F10',
         'E20000191107015320607E28', 'E20000191107015321907E60', 'E20000191107015324107EAC', 'E200001911070154209080CD',
         'E2000019110701542370805D',
         'E20000191107015326407F04', 'E20000191107015328307F60', 'E20000191107015329007F78', 'E200001911070154217080AD',
         'E20000191107015423108079']

    # 标签阵列宽与长
    WIDTH: int = 5
    HEIGHT: int = 5

    # 具体手势
    GESTURE: str = "c"
    GESTURE_TYPE: str = "letters"

    # 线性插值时间
    LINEAR_INTERPOLATION_TIME: int = 40
    FILTER_WINDOWS: int = 15
    BIO_RESOLUTION: int = 3

    GESTURE_DICT: dict = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "swipe_down_to_up": 6,
        "swipe_l_to_r": 7
    }

    # 划分训练集和测试集参数
    TEST_SIZE = 0.2
    RANDOM_SEED = 2023


class Path:

    # 原始RFID数据路径
    PATH_ORIGIN_DATA: str = r'E:\笨比J\RFID\Impinj R420\Data\原始数据\\'

    # 具体手势原始路径
    PATH_ORIGIN_GESTURE: str = PATH_ORIGIN_DATA + Config.GESTURE

    # 手势特征存放路径
    PATH_GESTURE_FEATURE: str = r'E:\笨比j\rfid\impinj r420\Data\Train\\' + \
                                Config.GESTURE_TYPE + r'\\' + Config.GESTURE

    # 手势原始特征图存放路径（未筛选）
    PATH_IMG_FEATURE: str = r'E:\笨比j\rfid\Impinj R420\img\\'

    # 数据聚合表路径
    PATH_TOGETHER: str = r'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\二十五标签_together.xls'
    PATH_PREPROCESSING: str = r'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\数据预处理.xls'

    # 训练数据存储路径
    PATH_DATA: str = r'E:\笨比J\RFID\Impinj R420\Data\data.csv'
    PATH_TRAIN: str = r'E:\笨比J\RFID\Impinj R420\Data\Train\train.csv'
    PATH_TEST: str = r'E:\笨比J\RFID\Impinj R420\Data\Test\test.csv'
