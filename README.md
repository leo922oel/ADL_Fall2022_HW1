# Homework 1 - Intent Classification & Slot Tagging
> Applied Deep Learning Fall 2022
## Shortcuts
- [Kaggle - Intent Classification](https://www.kaggle.com/competitions/intent-classification-ntu-adl-hw1-fall-2022)
- [Kaggle - Slot Tagging](https://www.kaggle.com/competitions/slot-tagging-ntu-adl-hw1-fall-2022)
## Environment
> python `3.9`

    # If you have conda, you can build a conda environment called "adl-hw1"
    make
    conda activate adl-hw1
    pip install -r requirement.txt

    # Otherwise
    pip install -r requirements.in

## Preprocessing
    bash preprocess.sh

## Intent cls
> default device is `CPU`, recommending use `GPU` with `--device cuda`

    # train
    python train_intent.py

    # test
    bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv

    ## or
    python test_intent.py --test_file /path/to/test.json --ckpt_path /path/to/pred.csv
## Slot tag
> default device is `CPU`, recommending use `GPU` with `--device cuda`

    # train
    python train_slot.py

    # test
    bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv

    ## or
    python test_slot.py --test_file /path/to/test.json --ckpt_path /path/to/pred.csv