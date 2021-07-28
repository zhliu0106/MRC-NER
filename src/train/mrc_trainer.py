import torch
import torch.nn as nn
from loguru import logger
import time
from tqdm import tqdm
from torch.nn.modules import BCEWithLogitsLoss
from train.trainer import Trainer
from train.metrics import QuerySpanF1
from typing import List, Dict


class MrcTrainer(Trainer):
    def __init__(self, model, config, accelerator):
        Trainer.__init__(self, model, config, accelerator)
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
        self.flat_ner = self.config["flat"]
        self.span_f1 = QuerySpanF1(flat=self.flat_ner)
        self.span_loss_candidates = self.config["span_loss_candidates"]

    def compute_loss(
        self,
        start_logits,
        end_logits,
        span_logits,
        start_labels,
        end_labels,
        match_labels,
        start_label_mask,
        end_label_mask,
    ):
        batch_size, seq_len = start_logits.size()
        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = (start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0) & (
                    end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0
                )
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len) & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len) & end_labels.unsqueeze(-2).expand(-1, seq_len, -1)),
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss

    def train(self, train_dataloader, dev_dataloader, test_dataloader, accelerator):
        num_iters_nochange = 0
        best_epoch = 0
        best_dev_pre, best_dev_rec, best_dev_f1 = -1, -1, -1
        num_epoch = self.config["num_epoch"]
        num_steps = len(train_dataloader)
        patience = self.config["patience"]
        self.scheduler = self.get_scheduler(train_dataloader)

        # 配置device
        self.model, self.optimizer, train_dataloader, dev_dataloader, test_dataloader = self.accelerator.prepare(self.model, self.optimizer, train_dataloader, dev_dataloader, test_dataloader)

        logger.info("Model Training ...")
        logger.info("Train Instances Size:%d" % len(train_dataloader.dataset))
        logger.info("  Dev Instances Size:%d" % len(dev_dataloader.dataset))

        for i in range(num_epoch):
            self.model.train()
            self.model.zero_grad()
            logger.info("=========Epoch: %d / %d=========" % (i + 1, num_epoch))

            start_time = time.time()
            total_loss = 0

            with tqdm(total=len(train_dataloader)) as pbar:
                pbar.set_description("train")
                for step, batch in enumerate(train_dataloader):
                    (
                        tokens,
                        token_type_ids,
                        start_labels,
                        end_labels,
                        start_label_mask,
                        end_label_mask,
                        match_labels,
                        sample_idx,
                        label_idx,
                    ) = batch

                    attention_mask = (tokens != 0).long()
                    start_logits, end_logits, span_logits = self.model(tokens, token_type_ids, attention_mask)
                    start_loss, end_loss, match_loss = self.compute_loss(
                        start_logits=start_logits,
                        end_logits=end_logits,
                        span_logits=span_logits,
                        start_labels=start_labels,
                        end_labels=end_labels,
                        match_labels=match_labels,
                        start_label_mask=start_label_mask,
                        end_label_mask=end_label_mask,
                    )

                    loss = start_loss + end_loss + match_loss
                    # loss.backward()
                    accelerator.backward(loss)
                    total_loss += loss.item()
                    batch_loss = loss.item()

                    if (step + 1) % self.config["grad_accum_steps"] == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_value"])
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.model.zero_grad()

                    if (step + 1) % self.config["logging_steps"] == 0:
                        log_lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"{i + 1} / {num_epoch} epoch - {step + 1} / {num_steps} step, lr: {log_lr}, loss: {batch_loss}"
                        )
                    pbar.update(1)

            end_time = time.time()
            logger.info(f"Epoch {i} consumes time: {end_time-start_time}")

            with torch.no_grad():
                self.model.eval()
                dev_all_result = self.eval(dev_dataloader)
                dev_avg_loss, dev_span_pre, dev_span_rec, dev_span_f1 = self.calculate_prf(dev_all_result)
                logger.info(f"Dev : p={dev_span_pre}, r={dev_span_rec}, f={dev_span_f1}, {dev_avg_loss=}")

                if dev_span_f1 >= best_dev_f1:
                    best_dev_pre, best_dev_rec, best_dev_f1 = dev_span_pre, dev_span_rec, dev_span_f1
                    best_epoch = i + 1
                    num_iters_nochange = 0
                    self.save_model()
                else:
                    num_iters_nochange += 1
                    if num_iters_nochange > patience:
                        break

        self.load_best_model()
        with torch.no_grad():
            self.model.eval()
            test_all_result = self.eval(test_dataloader)
            test_avg_loss, test_span_pre, test_span_rec, test_span_f1 = self.calculate_prf(test_all_result)

        logger.info(f"{'='*20} Best Epoch is {best_epoch} {'='*20}")
        logger.info(f" Dev: p={best_dev_pre}, r={best_dev_rec}, f={best_dev_f1}")
        logger.info(f"Test: p={test_span_pre}, r={test_span_rec}, f={test_span_f1}")

    def eval(self, dataloader):
        loss = 0
        outputs = list()

        with tqdm(total=len(dataloader)) as pbar:
            pbar.set_description("eval")
            for step, batch in enumerate(dataloader):
                (
                    tokens,
                    token_type_ids,
                    start_labels,
                    end_labels,
                    start_label_mask,
                    end_label_mask,
                    match_labels,
                    sample_idx,
                    label_idx,
                ) = batch
                attention_mask = (tokens != 0).long()
                start_logits, end_logits, span_logits = self.model(tokens, token_type_ids, attention_mask)
                start_logits, end_logits, span_logits = self.accelerator.gather(start_logits, end_logits, span_logits)
                start_labels, end_labels, match_labels, start_label_mask, end_label_mask = self.accelerator.gather(start_labels, end_labels, match_labels, start_label_mask, end_label_mask)
                start_loss, end_loss, match_loss = self.compute_loss(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    span_logits=span_logits,
                    start_labels=start_labels,
                    end_labels=end_labels,
                    match_labels=match_labels,
                    start_label_mask=start_label_mask,
                    end_label_mask=end_label_mask,
                )
                loss = start_loss + end_loss + match_loss
                output = dict()
                output["val_loss"] = loss
                output["start_loss"] = start_loss
                output["end_loss"] = end_loss
                output["match_loss"] = match_loss

                start_preds, end_preds = start_logits > 0, end_logits > 0
                span_f1_stats = self.span_f1(
                    start_preds=start_preds.cuda(),
                    end_preds=end_preds.cuda(),
                    match_logits=span_logits.cuda(),
                    start_label_mask=start_label_mask.cuda(),
                    end_label_mask=end_label_mask.cuda(),
                    match_labels=match_labels.cuda(),
                )
                output["span_f1_stats"] = span_f1_stats
                outputs.append(output)
                pbar.update(1)
            return outputs

    def calculate_prf(self, outputs: List[Dict]):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        all_counts = torch.stack([x["span_f1_stats"] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_rec = span_tp / (span_tp + span_fn + 1e-10)
        span_pre = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_pre * span_rec * 2 / (span_rec + span_pre + 1e-10)
        return avg_loss, span_pre, span_rec, span_f1
