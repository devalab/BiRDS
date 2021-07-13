from zipfile import ZipFile
import xtarfile
from argparse import ArgumentParser
import os
from glob import glob


def decompress_file(file, ext):
    print("Extracting file", file)
    folder = os.path.dirname(file)
    with xtarfile.open(file, "r:" + ext) as archive:
        archive.extractall(folder)


def decompress_dataset_files(dataset_dir):
    print("Extracting files from dataset", dataset_dir)
    for file in glob(os.path.join(dataset_dir, "*.tar.zst")):
        decompress_file(file, "zst")
    print()


def download_extract_data(hparams):
    # Download required data and extract
    os.system(
        "aria2c -c -x 8 -s 8 -d "
        + hparams.project_dir
        + " https://www.dropbox.com/s/cd9h2qtaphtvx6w/data.zip?dl=1"
    )

    print("Unzipping data.zip")
    with ZipFile(os.path.join(hparams.project_dir, "data.zip"), "r") as zip_ref:
        zip_ref.extractall(hparams.project_dir)

    decompress_dataset_files(os.path.join(hparams.data_dir, "scPDB"))
    decompress_dataset_files(os.path.join(hparams.data_dir, "2018_scPDB"))


def download_extract_model(hparams):
    os.system(
        "aria2c -c -x 8 -s 8 -d "
        + hparams.project_dir
        + " https://www.dropbox.com/s/1sfcam7tsggx4wm/model.tar.zst?dl=1"
    )
    decompress_file(os.path.join(hparams.project_dir, "model.tar.zst"), "zst")


def download_extract_prediction_files(hparams):
    os.system(
        "aria2c -c -x 8 -s 8 -d "
        + hparams.data_dir
        + " ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    )
    decompress_file(os.path.join(hparams.data_dir, "uniref50.fasta.gz"), "gz")

    os.system(
        "aria2c -c -x 8 -s 8 -d "
        + hparams.data_dir
        + " http://wwwuser.gwdg.de/~compbiol/uniclust/2017_10/uniclust30_2017_10_hhsuite.tar.gz"
    )
    decompress_file(
        os.path.join(hparams.data_dir, "uniclust30_2017_10_hhsuite.tar.gz"), "gz"
    )


def main(hparams):
    download_extract_data(hparams)
    download_extract_model(hparams)
    if hparams.predict:
        download_extract_prediction_files(hparams)


def parse_arguments():
    parser = ArgumentParser(
        description="Download files required for training and testing BiRDS",
        add_help=True,
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        type=str,
        help="Location of data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--predict",
        dest="predict",
        action="store_true",
        help="Download files required for prediction using BiRDS. Default: %(default)s",
    )
    parser.set_defaults(predict=False)
    hparams = parser.parse_args()
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    hparams.project_dir = os.path.dirname(hparams.data_dir)
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
