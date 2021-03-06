import os
import json
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from typing import List, Dict


def get_data_loader(config, prefix="train") -> DataLoader:
    assert prefix in ["train", "dev", "test"]

    if not os.path.exists(config[prefix]):
        logger.info(f"Loading {prefix} data SUCCESS")
        return None

    dataset = MrcNerDataset(
        config[prefix],
        tokenizer=BertTokenizer.from_pretrained(config["bert_dir"]),
        max_length=config["max_len"],
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True if prefix == "train" else False,
        collate_fn=collate_to_max_length_cuda if torch.cuda.is_available() else collate_to_max_length,
    )

    return dataloader


class MrcNerDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
    """

    def __init__(
        self, data_path: str, tokenizer: BertTokenizer, max_length: int = 128, possible_only=False, pad_to_maxlen=False
    ):
        self.all_data = json.load(open(data_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_maxlen = pad_to_maxlen

        if possible_only:
            self.all_data = [x for x in self.all_data if x["start_position"]]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index: int):
        """
        Args:
            index: int
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labels of NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id
        """
        data = self.all_data[index]
        tokens, type_ids = [], []
        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        end_positions = [x + 1 for x in end_positions]

        # ??????????????????
        query_context_tokens = ["[CLS]"]
        query_context_tokens.extend(list(query))
        query_context_tokens.append("[SEP]")
        query_context_tokens.extend(context.split(" "))
        query_context_tokens.append("[SEP]")

        # token 2 id
        for subword in query_context_tokens:
            if subword in ["[CLS]", "[SEP]"]:
                sub_tokens = [subword]
            else:
                sub_tokens = self.tokenizer.tokenize(subword)
                if len(sub_tokens) == 0:
                    sub_tokens = ["[UNK]"]
            token_idx = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            tokens += token_idx

        # [CLS]query[SEP] = 0, context[SEP] = 1
        type_ids.extend([0] * (len(query) + 2))
        type_ids.extend([1] * (len("".join(context.split(" "))) + 1))

        # ????????????????????????????????????
        queryOffset = [(0, 0)]
        for i in range(len(query)):
            queryOffset.append((i, i + 1))
        queryOffset.append((0, 0))
        contextOffset = []
        for i in range(len(context.split(" "))):
            contextOffset.append((i, i + 1))
        contextOffset.append((0, 0))
        queryOffset.extend(contextOffset)
        offsets = queryOffset

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        # context???tokens????????????
        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1) for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # ??????context??????token?????????mask???
        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0) for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0) for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is [SEP]
        sep_token = self.tokenizer.convert_tokens_to_ids("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[:-1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            label_idx,
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def collate_fn(data):
    batch = zip(*data)
    return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True) for x in batch])


def collate_fn_cuda(data):
    batch = zip(*data)
    return tuple([torch.tensor(x, device="cuda") if len(x[0].size()) < 1 else pad_sequence(x, True).cuda() for x in batch])


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)  # x[0]???tokens?????????
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))

    return output


def collate_to_max_length_cuda(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)  # x[0]???tokens?????????
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype, device="cuda")
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long, device="cuda")
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-2] for x in batch]).cuda())
    output.append(torch.stack([x[-1] for x in batch]).cuda())

    return output
