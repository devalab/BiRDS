import os
from argparse import ArgumentParser

# from test import load_nets_frozen


def get_all_sequences(file):
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def main(hparams):
    print(hparams)
    # sequences = get_all_sequences(file)
    # nets = load_nets_frozen(hparams)

    # TODO: Need to complete this


def parse_arguments():
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    parser.add_argument(
        "--ckpt_dir",
        default="../model",
        type=str,
        help="Checkpoint directory containing checkpoints of all 10 folds. Default: %(default)s",
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        type=str,
        help="Location of data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus to use for computation. Default: %(default)d",
    )
    parser.add_argument(
        "--file",
        default="../sample.predict",
        type=str,
        help="Location of the file containing 1 sequence per line. Default: %(default)s",
    )
    parser.set_defaults(validate=False)
    hparams = parser.parse_args()
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    hparams.ckpt_dir = os.path.abspath(os.path.expanduser(hparams.ckpt_dir))
    hparams.file = os.path.abspath(os.path.expanduser(hparams.file))
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
