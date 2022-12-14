import json
import pickle
import numpy as np
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.cnn_num_layers,
        args.rnn_num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.rnn_type,
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    ids = []
    tags = []
    lens = []

    for batch in dataloader:
        batch["tokens"] = batch["tokens"].to(args.device)
        batch["tags"] = batch["tags"].to(args.device)
        batch["mask"] = batch["mask"].to(args.device)

        with torch.no_grad():
            output = model(batch)
        ids += batch["id"]
        tags += output["pred_labels"].cpu().tolist()
        lens += batch["mask"].sum(-1).long().cpu().tolist()

    int_ids = [int(id[5:]) for id in ids]
    tags = [i for _, i in sorted(zip(int_ids, tags),)]
    lens = [i for _, i in sorted(zip(int_ids, lens),)]
    int_ids.sort()
    ids = [("test-"+str(id)) for id in int_ids]

    if args.pred_file.parent:
        args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for id, tag, len in zip(ids, tags, lens):
            f.write("%s," %(id))
            for idx, t in enumerate(tag):
                if idx < len-1:
                    f.write("%s " %(dataset.idx2label(t)))
                else:
                    f.write("%s\n" %(dataset.idx2label(t)))
                    break

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        # default="./ckpt/slot/",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--cnn_num_layers", type=int, default=2)
    parser.add_argument("--rnn_num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="LSTM")

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--seed", type=int, default=1000, help="seed for testing")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)