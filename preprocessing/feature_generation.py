# %%
# Some constants that will be required
# ALWAYS RUN THIS CODE CELL
import os

data_dir = os.path.abspath("../data/scPDB")
raw_dir = os.path.join(data_dir, "raw")
pssm_dir = os.path.join(data_dir, "pssm")
splits_dir = os.path.join(data_dir, "splits")
preprocessed_dir = os.path.join(data_dir, "preprocessed")

# %%
# Let us take all the amino acid properties available from the AAindex database
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2238890/
# The file AA_properties in the data folder contains all the values
# We will take only the features which are the least correlated as our features
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as cm

tmp_df = pd.read_csv("../data/AA_properties.csv", sep=",")
df = tmp_df.iloc[:, 7:].T


# Function used to normalize the values between 0 and 1
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# A way to view the correlation matrix
def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap("jet", 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title("Feature Correlation")
    labels = np.arange(0, len(df), 1)
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=np.arange(-1.1, 1.1, 0.1))
    plt.show()


# Using spearman correlation
corr = df.corr("spearman")
threshold = 0.6
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= threshold or corr.iloc[i, j] <= -threshold:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
features = df[selected_columns]

# Saving all the features and the selected features
normalize(df).to_csv("../data/all_features.csv")
normalize(features).to_csv("../data/selected_features.csv")


# %%
# Assuming that all the PSSMs have been copied to data/scPDB/pssm
# We can have more features included as well. For now, let us consider PSSMs
import csv
from collections import defaultdict

import numpy as np


# Get amino acid properties from the created files above
def get_amino_acid_properties(csv_file):
    feats = {}
    with open(csv_file) as f:
        records = csv.reader(f)
        for i, row in enumerate(records):
            if i == 0:
                length = len(row) - 1
                continue
            feats[row[0]] = np.array(
                [float(el) if el != "" else 0.0 for el in row[1:]], dtype=np.float32
            )
        feats["X"] = np.zeros(length)
    feats = defaultdict(lambda: np.zeros(length), feats)
    return feats


# List of amino acids and their integer representation
AA_ID_DICT = {
    "X": 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
}
AA_ID_DICT = defaultdict(lambda: 0, AA_ID_DICT)

# We generated PSSMs for select sequences
# Mapping different sequences to the one for which we generated
common_pssms = defaultdict(str)
with open(os.path.join(data_dir, "unique"), "r") as f:
    for line in f.readlines():
        line = line.strip().split()
        pssm_generated_for = line[0][:-1]
        common_pssms[pssm_generated_for] = pssm_generated_for
        for pdb_id_struct_chain in line[1:]:
            pdb_id_struct_chain = pdb_id_struct_chain[:-1]
            common_pssms[pdb_id_struct_chain] = pssm_generated_for


def get_pssm(pdb_id_struct, chain_id, length):
    pssm_pdb_id_struct, pssm_chain_id = common_pssms[
        pdb_id_struct + "/" + chain_id
    ].split("/")
    with open(
        os.path.join(pssm_dir, pssm_pdb_id_struct, pssm_chain_id + ".pssm"), "r"
    ) as f:
        lines = f.readlines()
    feature = np.zeros((21, length))
    for i, line in enumerate(lines):
        feature[i] = np.array(line.strip().split(), dtype=np.float32)
    return feature


# Amino acid physico-chemical features selected by removing highly correlated features
AA_sel_feats = get_amino_acid_properties(
    os.path.join(os.path.dirname(data_dir), "selected_features.csv")
)


# %%
# Now, let us preprocess the files again to generate the features directly that can be imported into pytorch easily
# For that we can define the generate_input function which can be used to generate various types of inputs
import numpy as np


# Without using distance map
feat_vec_len = 21 + 1 + 21 + len(AA_sel_feats["X"])


def generate_input(sample):
    X = np.zeros((feat_vec_len, sample["length"]))

    # One-hot encoding
    X[:21] = np.array([np.eye(21)[AA_ID_DICT[el]] for el in sample["sequence"]]).T

    # Positional encoding
    X[21] = np.arange(1, sample["length"] + 1, dtype=np.float32) / sample["length"]

    # PSSM
    X[22:43] = sample["pssm"]

    # AA Properties
    X[43:] = np.array([AA_sel_feats[aa] for aa in sample["sequence"]]).T

    return X


# Using the distance map to pick the closest 20 residues and creating features for each amino acid
# Amino acid no. + Distance from curr. aa + PSSM + AA Properties

# feat_vec_len = 21 + 1 + 21 + len(AA_sel_feats["X"])
# close_aa = 20


# def generate_input(sample):
#     X = np.zeros((feat_vec_len * close_aa, sample["length"]))
#     for i, aa_dist in enumerate(sample["ca_dist_map_true"]):
#         if sample["length"] <= close_aa:
#             indices = np.argsort(aa_dist)
#         else:
#             # Selects the smallest `close_aa` elements not in sorted order
#             indices = list(np.argpartition(aa_dist, close_aa)[:close_aa])
#             # Sort the indices according to their value
#             indices.sort(key=lambda x: aa_dist[x])
#         for j, idx in enumerate(indices):
#             if aa_dist[idx] == 1e6:
#                 # Implies we don't have structural info about any AA after this
#                 break
#             aa = sample["sequence"][idx]
#             X[j * feat_vec_len : j * feat_vec_len + 21, i] = np.eye(21)[AA_ID_DICT[aa]]
#             X[j * feat_vec_len + 21, i] = aa_dist[idx]
#             X[j * feat_vec_len + 22 : j * feat_vec_len + 43, i] = sample["pssm"][:, idx]
#             X[j * feat_vec_len + 43 : (j + 1) * feat_vec_len, i] = np.array(
#                 AA_sel_feats[aa]
#             )
#     return X


# With an inverted distance map
# This does not work very well

# def generate_input(sample):
#     X = np.zeros((feat_vec_len, sample["length"]))

#     # One-hot encoding
#     X[:21] = np.array([np.eye(21)[AA_ID_DICT[el]] for el in sample["sequence"]]).T

#     # PSSM
#     X[22:43] = sample["pssm"]

#     # AA Properties
#     X[43:] = np.array([AA_sel_feats[aa] for aa in sample["sequence"]]).T

#     # Invert the distance map and matrix multiply with X so that we get a combination of all features
#     inverted_dist = 1 / sample["ca_dist_map_true"]
#     np.fill_diagonal(inverted_dist, 1.0)
#     X_T = inverted_dist.dot(X.T)
#     for i in range(sample["length"]):
#         X_T[i] = X_T[i] / np.sum(inverted_dist[i])

#     X = X_T.T

#     # Positional encoding
#     X[21] = np.arange(1, sample["length"] + 1, dtype=np.float32) / sample["length"]

#     return X


# %%
# USING CONCATENATION STRATEGY

for pdb_id_struct in sorted(os.listdir(preprocessed_dir)):
    flg = True
    pre = os.path.join(preprocessed_dir, pdb_id_struct)
    features_file = os.path.join(pre, "features.npy")
    labels_file = os.path.join(pre, "labels.npy")

    if os.path.exists(labels_file):
        continue
    print(pdb_id_struct)

    for file in sorted(os.listdir(pre)):
        # In case features were generated but not labels, redo it
        if file == "features.npy":
            continue
        chain_id = file[-len(".npz") - 1 : -len(".npz")]
        sample = np.load(os.path.join(pre, file))
        sample = {
            key: sample[key].item() if sample[key].shape == () else sample[key]
            for key in sample
        }
        sample["pssm"] = get_pssm(pdb_id_struct, chain_id, sample["length"])
        if flg:
            X = generate_input(sample)
            y = sample["labels"]
            flg = False
        else:
            # Using concatenation strategy
            tmp = generate_input(sample)
            X = np.concatenate((X, tmp), 1)
            y = np.concatenate((y, sample["labels"]), 0)

    np.save(features_file, X)
    np.save(labels_file, y)


# SAVING ALL CHAINS AS DIFFERENT PROTEINS
# for pdb_id_struct in sorted(os.listdir(preprocessed_dir)):
#     pre = os.path.join(preprocessed_dir, pdb_id_struct)
#     print(pdb_id_struct)

#     for file in sorted(os.listdir(pre)):
#         # In case features were generated but not labels, redo it
#         if not file.endswith(".npz"):
#             continue
#         chain_id = file[-len(".npz") - 1 : -len(".npz")]
#         features_file = os.path.join(pre, "features" + chain_id + ".npy")
#         labels_file = os.path.join(pre, "labels" + chain_id + ".npy")
#         if os.path.exists(features_file) and os.path.exists(labels_file):
#             continue
#         sample = np.load(os.path.join(pre, file))
#         sample = {
#             key: sample[key].item() if sample[key].shape is () else sample[key]
#             for key in sample
#         }
#         sample["pssm"] = get_pssm(pdb_id_struct, chain_id, sample["length"])
#         X = generate_input(sample)
#         y = sample["labels"]

#         np.save(features_file, X)
#         np.save(labels_file, y)


# %%
# Let us save the above files and store in a safe space


def archive_dir(folder, pattern, name):
    parent_dir = os.path.dirname(folder)
    folder = os.path.basename(folder)
    #     print(parent_dir, folder)
    if os.path.exists(os.path.join(parent_dir, name)):
        print("Warning:", name, "already exists in", parent_dir)
        inp = input("Do you want to overwrite existing file? (y/n): ")
        if inp[0].lower() == "n":
            return
    # Using only a single ! command since multiple ! spawn different bash shells
    # For some reason, the below code is doubling the contents of the tar file
    # !cd $parent_dir; find $folder -name "$pattern" | tar --sort=name -I zstd -cf $name -T -; rsync -avP $name crvineeth97@ada:/share2/crvineeth97/compressed/scPDB; cd -; # noqa: E501
    # To untar, use
    # !tar -I zstd -xvf $name


# Examples:
# archive_dir(raw_dir, "*", "raw.tar.zst")
# archive_dir(preprocessed_dir, "*.npy", "features_labels.tar.zst")
# archive_dir(preprocessed_dir, "*.npz", "preprocessed_chains.tar.zst")
# archive_dir(pssm_dir, "*", "pssm.tar.zst")
