# Protein Binding Site Prediction

## Environment

The project can be setup using [conda](https://docs.conda.io/en/latest/miniconda.html).
Run the following commands in the root folder of the project after cloning it

```bash
conda env create -f environment.yml
python -m pip install -e .
conda activate birds
```

## Required files

### Training/Testing

Download the following files into the root folder of the project

- The files required by the model can be downloaded from [primary](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ravindrachelur_v_research_iiit_ac_in/EV4k56vFnuxArB81zNIFfzgBU9t15ajDwrfQrBW7RNiT7A?e=m6NRfJ), [backup](https://www.dropbox.com/s/cd9h2qtaphtvx6w/data.zip)

- The models that were trained and used in the paper can be downloaded from [primary](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ravindrachelur_v_research_iiit_ac_in/EZgET08RIBhFl3ut2YoM4x8BMPr0wz-LYj7IB8IHsZM40w?e=97FU59), [backup](https://www.dropbox.com/s/1sfcam7tsggx4wm/models.tar.zst)

Run the following commands in the root folder of the project.

```bash
unzip data.zip
cd data/scPDB
for file in *.tar.zst; do tar -I zstd -xf "$file"; done
cd -

cd data/2018_scPDB
for file in *.tar.zst; do tar -I zstd -xf "$file"; done
cd -

tar -I zstd -xvf model.tar.zst
```

### Predictions

uniref50.fasta and uniclust30_2017_10_hhsuite are needed for the generation of MSAs

```bash
aria2c -c -x 8 -s 8 -d "./data/" ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
aria2c -c -x 8 -s 8 -d "./data/" http://wwwuser.gwdg.de/~compbiol/uniclust/2017_10/uniclust30_2017_10_hhsuite.tar.gz
tar xvzf ./data/uniref50.fasta.gz -C ./data/
tar xvzf ./data/uniclust30_2017_10_hhsuite.tar.gz -C ./data/
./msa_generator/hhsuite2/bin/esl-sfetch --index ./data/uniref50.fasta
```

## Training

To train a model for a particular fold, simply run

```bash
python train.py
```

To view all the supported options for training

```bash
python train.py --help
```

## Testing

To test the models

```bash
python test.py
```

To run on the validation sets

```bash
python test.py --validate
```

## Predicting

For predictions, the following files are required
