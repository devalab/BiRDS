import os
import re
from argparse import ArgumentParser
from shutil import copyfile
from test import *

from msa_generator.extract_features import extract_features_from_file
from msa_generator.generate_msa import generate_msas_from_file

from utils import *

PIS = "[a-zA-Z0-9]{4}_[0-9]{1}"


def main(hparams):
    print(hparams)
    raw_dir = os.path.join(hparams.dataset_dir, "raw")
    splits_dir = os.path.join(hparams.dataset_dir, "splits")
    for pis in os.listdir(raw_dir):
        pre = os.path.join(raw_dir, pis)
        assert os.path.isdir(pre), pre + " should be a directory"
        assert re.match(PIS, pis), pis + " should match the regex " + PIS
        sequence_fasta = os.path.join(pre, "sequence.fasta")
        assert os.path.exists(sequence_fasta) and os.path.isfile(
            sequence_fasta
        ), "sequence.fasta is not present or is not a file"
        create_chains_from_sequence_fasta(pre, sequence_fasta)

    unique = os.path.join(splits_dir, "unique")
    make_unique_file(raw_dir, splits_dir)
    generate_msas_from_file(hparams.dataset_dir, unique, hparams.cpus)
    extract_features_from_file(hparams.dataset_dir, unique, hparams.cpus)
    store_features_as_numpy(hparams.dataset_dir, unique)
    create_info_file(hparams.dataset_dir)
    copyfile(unique, os.path.join(hparams.dataset_dir, "unique"))

    hparams.data_dir = os.path.dirname(hparams.dataset_dir)
    hparams.predict = True
    nets = load_nets_frozen(hparams)

    print("Running models on the predict set")
    test_dl = nets[0].test_dataloader()
    device = nets[0].device
    predictions = []
    for idx, batch in tqdm(enumerate(test_dl)):
        data, meta = move_data_to_device(batch, device)
        y_pred = predict(nets, data, meta).cpu().numpy()
        batch_size = len(meta["length"])
        for i in range(batch_size):
            piscs = meta["pisc"][i]
            sequences = meta["sequence"][i]
            cumulative = 0
            for j in range(len(piscs)):
                label = "".join(
                    [
                        "1" if el else "0"
                        for el in y_pred[i][cumulative : cumulative + len(sequences[j])]
                    ]
                )
                cumulative += len(sequences[j])
                predictions.append([piscs[j], sequences[j], label])
            predictions.append([])
    # predictions = ["\t".join(line) + "\n" for line in predictions]
    predictions_file = os.path.join(hparams.dataset_dir, "predictions.txt")
    with open(predictions_file, "w") as f:
        for prediction in predictions:
            if prediction == []:
                f.write("\n")
                continue
            for el in prediction:
                f.write(el + "\n")
    print("Predictions written to", predictions_file)


def parse_arguments():
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    parser.add_argument(
        "--ckpt_dir",
        default="../model",
        type=str,
        help="Checkpoint directory containing checkpoints of all 10 folds. Default: %(default)s",
    )
    parser.add_argument(
        "--dataset-dir",
        default="../data/predict",
        type=str,
        help="Location of the fasta files which need to be predicted. \
                Format XXXX_Y/Z.fasta \
                    XXXX is any random 4 alphanumeric directory\
                        Y is the structure number\
                            Z is a random alphanumeric for the chain \
                            Default: %(default)s",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus to use for model prediction. Default: %(default)d",
    )
    parser.add_argument(
        "--cpus",
        default=24,
        type=int,
        help="Number of cpus to use for MSA generation. Default: %(default)d",
    )
    parser.set_defaults(validate=False)
    hparams = parser.parse_args()
    hparams.dataset_dir = os.path.abspath(os.path.expanduser(hparams.dataset_dir))
    hparams.ckpt_dir = os.path.abspath(os.path.expanduser(hparams.ckpt_dir))
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
