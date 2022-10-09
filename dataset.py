from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        sorted_samples = samples.sort(key=lambda s: len(s["text"].split()), )
        batch = {}
        batch["text"] = [s["text"].split() for s in sorted_samples]
        batch["len"] = torch.tensor([min(len(s), self.max_len) for s in batch["text"]])
        batch["text"] = torch.tensor(
            self.vocab.encode_batch(batch["text"], self.max_len)
            )
        batch["id"] = torch.tensor([s["id"] for s in sorted_samples])
        if "intent" in sorted_samples.keys():
            batch["intent"] = torch.tensor(
                self.label2idx(s["intent"] for s in sorted_samples)
            )
        else: 
            batch["intent"] = torch.zeros(len(sorted_samples), dtype=torch.uint8)

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def count_label(self):
        self.count = torch.zeros(len(self.label_mapping))
        if "tags" in self.data:
            for data in self.data:
                for tag in data["tags"]:
                    self.count[self.label2idx(tag)] += 1
        else: self.count += 1

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        sorted_samples = samples.sort(key=lambda s: len(s["tokens"]), )
        batch = {}
        batch["tokens"] = [s["tokens"] for s in sorted_samples]
        batch["len"] = torch.tensor([min(len(s), self.max_len) for s in batch["tokens"]])
        batch["tokens"] = torch.tensor(
            self.vocab.encode_batch(batch["tokens"], self.max_len)
            )
        batch["id"] = torch.tensor([s["id"] for s in sorted_samples])
        if "tags" in sorted_samples.keys():
            batch["tags"] = [[self.label2idx(t) for t in s["tags"]] for s in sorted_samples]
            batch["tags"] = pad_to_len(batch["tags"], self.max_len, 0)
        else: 
            batch["tags"] = [[0]*self.max_len] * len(sorted_samples)
        batch["tags"] = torch.tensor(batch["tags"])
        # batch["mask"] = batch["tokens"].gt(0).float()

        return batch