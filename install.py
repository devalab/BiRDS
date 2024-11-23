import gzip
import os
import shutil
import sys
import tarfile
from argparse import ArgumentParser
from glob import glob
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

import zstandard as zstd


def decompress_file(file, ext):
    print("Extracting file", file)
    folder = os.path.dirname(file)
    if ext == "zst":
        with open(file, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as archive:
                    archive.extractall(folder)
    elif ext == "gz":
        # Check if it's a tar.gz or just a gzip file
        try:
            with tarfile.open(file, "r:gz") as archive:
                archive.extractall(folder)
        except tarfile.ReadError:
            # Handle plain gzip files
            decompressed_file = os.path.splitext(file)[0]
            with gzip.open(file, 'rb') as f_in:
                with open(decompressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed to {decompressed_file}")
    else:
        with tarfile.open(file, "r:" + ext) as archive:
            archive.extractall(folder)


def decompress_dataset_files(dataset_dir):
    print("Extracting files from dataset", dataset_dir)
    for file in glob(os.path.join(dataset_dir, "*.tar.zst")):
        decompress_file(file, "zst")
    print()


def download_file(dir, url):
    filename = url.split("/")[-1].split("?")[0]
    filepath = os.path.join(dir, filename)
    if not os.path.exists(filepath):
        print("Downloading", filename)
        try:
            with urlopen(url) as response, open(filepath, "wb") as out_file:
                # Get total file size from headers
                total_length = response.info().get("Content-Length")
                if total_length is None:
                    # Write file directly if no length header
                    out_file.write(response.read())
                else:
                    # Stream and show progress bar
                    dl = 0
                    total_length = int(total_length)
                    while True:
                        chunk = response.read(4096)
                        if not chunk:
                            break
                        dl += len(chunk)
                        out_file.write(chunk)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                        sys.stdout.flush()
            print()  # Add a newline after the progress bar
        except URLError as e:
            print(f"Failed to download {url}: {e}")
    return filepath


def download_extract_data(hparams):
    # Download required data and extract
    download_file(
        hparams.project_dir, "https://www.dropbox.com/s/cd9h2qtaphtvx6w/data.zip?dl=1"
    )

    print("Unzipping data.zip")
    with ZipFile(os.path.join(hparams.project_dir, "data.zip"), "r") as zip_ref:
        zip_ref.extractall(hparams.project_dir)

    decompress_dataset_files(os.path.join(hparams.data_dir, "scPDB"))
    decompress_dataset_files(os.path.join(hparams.data_dir, "2018_scPDB"))


def download_extract_model(hparams):
    download_file(
        hparams.project_dir,
        "https://www.dropbox.com/s/1sfcam7tsggx4wm/model.tar.zst?dl=1",
    )
    decompress_file(os.path.join(hparams.project_dir, "model.tar.zst"), "zst")


def download_extract_prediction_files(hparams):
    download_file(
        hparams.data_dir,
        "ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
    )
    decompress_file(os.path.join(hparams.data_dir, "uniref50.fasta.gz"), "gz")

    download_file(
        hparams.data_dir,
        "http://wwwuser.gwdg.de/~compbiol/uniclust/2017_10/uniclust30_2017_10_hhsuite.tar.gz",
    )
    decompress_file(
        os.path.join(hparams.data_dir, "uniclust30_2017_10_hhsuite.tar.gz"), "gz"
    )


def download_extract_visualization_files(hparams):
    test_dir = os.path.join(hparams.data_dir, "2018_scPDB")
    download_file(
        test_dir,
        "https://www.dropbox.com/s/b0qoes4bjdnh9m3/sc6k_visualize.tar.zst?dl=1",
    )
    decompress_file(os.path.join(test_dir, "sc6k_visualize.tar.zst"), "zst")


def main(hparams):
    download_extract_data(hparams)
    download_extract_model(hparams)
    if hparams.predict:
        download_extract_prediction_files(hparams)
    if hparams.visualize:
        download_extract_visualization_files(hparams)


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
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Download files required for visualization of the outputs on the test set. Default: %(default)s",
    )
    parser.set_defaults(visualize=False)
    hparams = parser.parse_args()
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    hparams.project_dir = os.path.dirname(hparams.data_dir)
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
