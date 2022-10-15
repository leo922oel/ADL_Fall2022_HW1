import json
from argparse import ArgumentParser
import numpy as np

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def main(args):
    with open(args.eval_json) as f:
        eval_data = json.load(f)
        ground = [d["tags"] for d in eval_data]

    with open(args.predict_csv) as f:
        pred = [line.split(",")[1].split() for line in f.readlines()[1:]]

    print(classification_report(ground, pred, scheme=IOB2, mode="strict"))

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--eval_json", default="data/slot/eval.json")
    parser.add_argument("--predict_csv", default="./eval.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())