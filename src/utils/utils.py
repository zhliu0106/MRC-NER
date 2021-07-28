import os
import time
import json
import pickle
import torch
import random
import numpy as np
from typing import Dict
from loguru import logger

def write_json(file: str, obj: Dict) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, ensure_ascii=False)


def read_json(file: str) -> Dict:
    with open(file, "r") as f:
        data = json.load(f)
    return data


def read_pkl(file: str) -> Dict:
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def get_log_file_name(path: str):
    """
    根据当前时间生成日志名
    """
    currentTime = time.strftime("%Y-%m-%d_%H")
    log_file_name = os.path.join(path, currentTime + ".log")
    return log_file_name


def init_logger(path: str):
    """
    初始化logger
    """
    log_file_name = get_log_file_name(path)
    logger.add(log_file_name, backtrace=True, diagnose=True)

def set_seed(seed_num: int) -> None:
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


def show_config(config: Dict) -> None:
    print("========Here is your configuration========")
    for key, value in config.items():
        print(f"\t{key} = {value}")
