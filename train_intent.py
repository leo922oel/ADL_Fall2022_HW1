import json
import pickle
import random
from turtle import color
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, CEMetrics, IntentMetrics

from seqeval.metrics import classification_report

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_epoch(args, model, dataloader, optim):
    model.train()
    ce = CEMetrics()
    acc = IntentMetrics()

    bar = tqdm(dataloader)
    for i, batch in enumerate(bar):
        batch["text"] = batch["text"].to(args.device)
        batch["intent"] = batch["intent"].to(args.device)

        optim.zero_grad()
        output = model(batch)

        bar.set_postfix(loss=output["loss"].item(), iter=i, lr=optim.param_groups[0]["lr"])

        ce.udpate(output["loss"], batch["intent"].size(0))
        acc.update(batch["intent"].detach().cpu(), output["pred_labels"].detach().cpu())
        loss = output["loss"]

        loss.backward()
        optim.step()

    acc.eval()
    print("Train loss: {:6.4f}\t Acc: {:6.4f}".format(ce.avg, acc.acc))
    return ce.avg, acc.acc

@torch.no_grad()
def validation(args, model, dataloader, ):
    model.eval()
    ce = CEMetrics()
    acc = IntentMetrics()

    for batch in dataloader:
        batch["text"] = batch["text"].to(args.device)
        batch["intent"] = batch["intent"].to(args.device)

        output = model(batch)

        ce.udpate(output["loss"], batch["intent"].size(0))
        acc.update(batch["intent"].detach().cpu(), output["pred_labels"].detach().cpu())


    acc.eval()
    print("Valid loss: {:6.4f}\t Acc: {:6.4f}".format(ce.avg, acc.acc))
    return ce.avg, acc.acc

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir / f"{args.name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloader: Dict[str, DataLoader] = {
        split: DataLoader(split_data, args.batch_size, shuffle=True, collate_fn=split_data.collate_fn)
        for split, split_data in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers,
            args.dropout, args.bidirectional, datasets[TRAIN].num_classes, args.rnn_type).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_iter = 0
    train_acc_list = np.zeros(args.num_epoch)
    val_acc_list = np.zeros(args.num_epoch)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        print(f"Epoch: {epoch}")
        train_ce, train_acc = train_epoch(args, model, dataloader[TRAIN], optimizer, )
        val_ce, val_acc = validation(args, model, dataloader[DEV], )
        print(f"the best model ACC -- {best_acc}, epoch: {best_iter}")

        ckpt_path = f"{ckpt_dir}/{epoch+1}-model.pth"
        best_path = f"{ckpt_dir}/best-model.pth"
        torch.save(model.state_dict(), ckpt_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_iter = epoch
            torch.save(model.state_dict(), best_path)
            print(f"Save model checkpoint -- {best_path}")
    # TODO: Inference on test set

        train_acc_list[epoch] = train_acc
        val_acc_list[epoch] = val_acc
    """
    plt.plot(train_acc_list, color="red", )
    plt.plot(val_acc_list, color="blue", )
    plt.title(f"{args.num_epoch} epochs with batch size={args.batch_size}")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend(("train", "valid"))
    plt.savefig(f"./{args.batch_size}.png")
    """

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument("--name", type=str, default="1013", )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="LSTM")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000, help="seed for training")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
