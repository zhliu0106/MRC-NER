import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import write_json
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.set_optimizer()

    def set_optimizer(self):
        optimizer_name = self.config["optimizer"].lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-8)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["lr"], momentum=0.9, weight_decay=1e-8)
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-8)
        elif optimizer_name == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-8)
        else:
            logger.info("Wrong Optimizer!")

    def get_scheduler(self, dataloader):
        total_steps = len(dataloader) * self.config["num_epoch"] / self.config["grad_accum_steps"]
        warmup_steps = total_steps * self.config["warmup"]
        scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        return scheduler

    def save_model(self):
        final_model = self.config["save_model_dir"]
        if not os.path.exists(final_model):
            os.makedirs(final_model)

        model_info = self.model.get_model_info()
        logger.info(f"Saving model to: {final_model}")
        torch.save(self.model.state_dict(), os.path.join(final_model, "model.bin"))
        write_json(os.path.join(final_model, "config.json"), model_info)

    def load_best_model(self):
        device = next(self.model.parameters()).device
        model_params_path = os.path.join(self.config["save_model_dir"], "model.bin")
        self.model.load_state_dict(torch.load(model_params_path, map_location=device))

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
