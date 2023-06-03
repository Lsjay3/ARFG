# -*- coding:utf-8 -*-
import os
from pathlib import Path


def create_directory_if_not_exist(path: str):
    """
    检查给定的文件或文件夹路径是否存在，若不存在则创建（对文件路径仅会创建至其父目录）
    :type path: str
    :param path: 给定的文件或文件夹路径
    :return:
    """
    path_info = Path(path)
    if not path_info.suffix == '':
        path_info = path_info.parent
    if not path_info.exists():
        path_info.mkdir(parents=True, exist_ok=True)
