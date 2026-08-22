"""Configuration, metric, and logging helpers for the KS experiment."""

import logging
import os
from datetime import datetime
from os import path as osp


def deep_update_dict(source, target):
    """Recursively merge source values into target."""
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            deep_update_dict(value, child)
        else:
            target[key] = value
    return target


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, value):
        self.v = (self.v * self.n + value) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def create_logger(cfg, rank=0, test=False):
    dataset = "debug" if cfg.get("debug") else cfg["dataset"]["dataset_name"]
    output_dir = cfg.get("output_dir", "")

    if test:
        log_dir = osp.join(output_dir, dataset, "test")
        exp_id = cfg.get("test", {}).get("exp_id") or datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S-%f"
        )
    else:
        log_dir = osp.join(output_dir, dataset, "logs")
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        backbone = f"{cfg['visual']['name']}_{cfg['text']['name']}"
        loss = cfg["loss"]["type"]
        head = cfg.get("head", {}).get("type", "head")
        exp_id = f"{dataset}_{backbone}_{loss}_{cfg['seed']}_{head}_{time_str}"

    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, f"{exp_id}.log")

    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)-15s %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file, exp_id
