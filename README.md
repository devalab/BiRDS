# Protein Binding Site Prediction

## Environment

The project can be setup using [conda](https://docs.conda.io/en/latest/miniconda.html).
Run the following commands in the root folder of the project after cloning it

```bash
conda env create -f environment.yml
conda activate birds
python -m pip install -e .
```

## Required files

Run install.py script to download the required files and automatically extract them as well

```bash
python install.py
```

In case the downloads are not successful, please download the following files to the root directory of the project and then run the script

- The files required by the model for training/testing can be downloaded from [primary](https://www.dropbox.com/s/cd9h2qtaphtvx6w/data.zip), [backup](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ravindrachelur_v_research_iiit_ac_in/EV4k56vFnuxArB81zNIFfzgBU9t15ajDwrfQrBW7RNiT7A?e=m6NRfJ)

- The models that were trained and used in the paper can be downloaded from [primary](https://www.dropbox.com/s/1sfcam7tsggx4wm/model.tar.zst), [backup](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ravindrachelur_v_research_iiit_ac_in/EZgET08RIBhFl3ut2YoM4x8BMPr0wz-LYj7IB8IHsZM40w?e=cp9TmS)

## Training

```bash
cd birds
# View all the supported options for training
python train.py --help

# Train the default model (with parameters) used in the paper for fold 0 
python train.py --fold 0
```

## Testing

```bash
cd birds
# Test the models
python test.py

# Run the models on their validation sets
python test.py --validate
```

## Predicting

uniref50.fasta and uniclust30_2017_10_hhsuite are needed for the generation of MSAs

```bash
python install.py --predict
```

For predictions, the following format has to be followed. Generate a random 4 alphanumeric character PDB code, 1 character structure number for your protein sequence. Let's assume that the PDB code is abde and the structure number is 1. Then the directory and sequence file need to be created as follows

```bash
mkdir -p ./data/predict/raw/abde_1/
touch ./data/predict/raw/abde_1/sequence.fasta
```

In sequence.fasta, put the sequence in the format shown below

```fasta
>ABDE:A|PDBID|CHAIN|SEQUENCE
NSELDRLSKDDRNWVMQTKDYSATHFSRLTEINSHNVKNLKVAWTLSTGTLHGHEGAPLVVDGIMYIHTPFPNNVYAVDLNDTRKMLWQYKPKQNPAARAVACCDVVNRGLAYVPAGEHGPAKIFLNQLDGHIVALNAKTGEEIWKMENSDIAMGSTLTGAPFVVKDKVLVGSAGAELGVRGYVTAYNIKDGKQEWRAYATGPDEDLLLDKDFNKDNPHYGQFGLGLSTWEGDAWKIGGGTNWGWYAYDPKLDMIYYGSGNPAPWNETMRPGDNKWTMTIWGRDADTGRAKFGYQKTPHDEWDYAGVNYMGLSEQEVDGKLTPLLTHPDRNGLVYTLNRETGALVNAFKIDDTVNWVKKVDLKTGLPIRDPEYSTRMDHNAKGICPSAMGYHNQGIESYDPDKKLFFMGVNHICMDWEPFMLPYRAGQFFVGATLNMYPGPKGMLGQVKAMNAVTGKMEWEVPEKFAVWGGTLATAGDLVFYGTLDGFIKARDTRTGELKWQFQLPSGVIGHPITYQHNGKQYIAIYSGVGGWPGVGLVFDLKDPTAGLGAVGAFRELAHYTQMGGSVFVFSL
>ABDE:B|PDBID|CHAIN|SEQUENCE
YDGTHCKAPGNCWEPKPGYPDKVAGSKYDPKHDPNELNKQAESIKAMEARNQKRVENYAKTGKFVYKVEDIK
```

Please note that the predictions will take time since they are dependent on the generation of MSAs. There is verbose logging and some speed up optimizations. In case it is taking too long. Please follow the instructions in msa_generator for generating MSAs for a lot of sequences

```bash
cd birds
python predict.py
```

## Visualize

To visualize the results on the test set

```bash
python install.py --visualize
cd birds/visualize
python visualize.py
```
