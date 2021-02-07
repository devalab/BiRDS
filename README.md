# Protein Binding Site Prediction

## Environment

The project can be setup using conda. Run the following commands in the root folder of the project.

```bash
conda env create -f environment.yml
python -m pip install -e .
```

## Data

The data can be downloaded from [here](https://www.dropbox.com/s/cd9h2qtaphtvx6w/data.zip?dl=1)

Run the following commands in the root folder of the project.

```bash
unzip data.zip
cd data/scPDB
for file in *.tar.zst; do tar -I zstd -xf "$file"; done
cd -

cd data/2018_scPDB
for file in *.tar.zst; do tar -I zstd -xf "$file"; done
cd -

cd pbsp
```

