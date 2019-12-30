from os import environ

from torch import device
from torch.cuda import is_available

# List of amino acids and their integer representation
AA_ID_DICT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}

# Deletes already preprocessed data in and uses the raw data
# to regenerate the preprocessed data
FORCE_PREPROCESSING_OVERWRITE = False

PROJECT_FOLDER = environ["PWD"]

# Which device to use for tensor computations
DEVICE = device("cpu")
if is_available():
    print("CUDA is available. Using GPU")
    DEVICE = device("cuda")
