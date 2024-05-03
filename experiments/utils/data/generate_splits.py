from argparse import ArgumentParser
import json
import logging

from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from experiments.utils import set_logger
set_logger()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--max-models", default=None, type=int)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    save_path = Path(args.save_path)
    data_split = defaultdict(list)
    all_files = [p.as_posix() for p in data_root.glob("**/*.pth")]
    if args.max_models is not None:
        all_files = all_files[:args.max_models]

    # test split
    train_files, test_files = train_test_split(all_files, test_size=args.test_size)
    data_split["test"] = test_files

    # val split
    train_files, val_files = train_test_split(train_files, test_size=args.val_size)
    data_split["val"] = val_files

    data_split["train"] = train_files

    logging.info(f"train size: {len(data_split['train'])}, "
                 f"val size: {len(data_split['val'])}, test size: {len(data_split['test'])}")

    with open(save_path, "w") as file:
        json.dump(data_split, file)
