# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from collections import defaultdict
from glob import glob

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from tqdm import tqdm

AMINO_ACIDS = "XACDEFGHIKLMNPQRSTVWY"
AA_DICT = defaultdict(lambda: 0, {aa: idx for idx, aa in enumerate(AMINO_ACIDS)})


class scPDB(Dataset):
    def __init__(self, hparams, test=False):
        super().__init__()
        self.hparams = hparams
        self.test = test
        if test:
            self.dataset_dir = os.path.join(hparams.data_dir, "2018_scPDB")
        else:
            self.dataset_dir = os.path.join(hparams.data_dir, "scPDB")
            self.splits_dir = os.path.join(self.dataset_dir, "splits")
            self.fold = str(hparams.fold)
        self.raw_dir = os.path.join(self.dataset_dir, "raw")
        self.preprocessed_dir = os.path.join(self.dataset_dir, "preprocessed")
        self.msa_dir = os.path.join(self.dataset_dir, "msa")

        # Get features and labels that are small in size
        self.sequences, self.labels = self.get_sequences_and_labels()
        # if hparams.use_cb_coords:
        #     self.coords = self.get_npy("cb_coords", True)
        # else:
        #     self.coords = self.get_npy("ca_coords", True)
        self.aap = self.get_npy("aap", hparams.use_aap)
        self.pssm = self.get_npy("pssm", hparams.use_pssm)
        self.ss2 = self.get_npy("ss2", hparams.use_ss2)
        self.solv = self.get_npy("solv", hparams.use_solv)
        self.spot_ss3 = self.get_npy("ss3", hparams.use_ss3)
        self.spot_ss8 = self.get_npy("ss8", hparams.use_ss8)
        self.spot_asa = self.get_npy("asa", hparams.use_asa)
        self.spot_hse = self.get_npy("hse", hparams.use_hse)
        self.spot_cn13 = self.get_npy("cn13", hparams.use_cn13)
        self.spot_torsion = self.get_npy("torsion", hparams.use_torsion)

        self.available_data = sorted(list(self.labels.keys()))
        # ------------MAPPINGS------------
        # pdbID to pdbID_structure mapping
        self.pi_to_pis = defaultdict(list)
        for key in self.available_data:
            pis = key.split("/")[0]
            if pis not in self.pi_to_pis[key[:4]]:
                self.pi_to_pis[key[:4]].append(pis)

        # pdbID_structure TO pdbID_structure/chain mapping
        self.pis_to_pisc = defaultdict(list)
        for key in self.available_data:
            self.pis_to_pisc[key.split("/")[0]].append(key)

        # pdbID_structure/chain to available MSA pdbID_structure/chain sequence
        self.pisc_to_mpisc = {}
        if test:
            unique = "no_one_msa_unique"
        else:
            unique = "unique"
        with open(os.path.join(self.dataset_dir, unique), "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                mpisc = line[0][:-1]
                self.pisc_to_mpisc[mpisc] = mpisc
                for pisc in line[1:]:
                    self.pisc_to_mpisc[pisc[:-1]] = mpisc

        # Dataset pdbID to index mapping
        if test:
            self.dataset = sorted(list(self.pi_to_pis.keys()))
        else:
            self.train_fold, self.valid_fold = self.get_fold()
            self.dataset = sorted(self.train_fold + self.valid_fold)

        self.pi_to_index = {pi: idx for idx, pi in enumerate(self.dataset)}

        if not test:
            if hparams.pos_weight:
                print("Using provided positional weighting")
                self.pos_weight = [hparams.pos_weight]
            elif hparams.pos_weight == 0.0:
                print("Precomputing positional weights...")
                self.pos_weight = self.compute_pos_weight()
            else:
                print("Positional weights will be computed on the fly")
                self.pos_weight = None

            self.train_indices = [self.pi_to_index[pi] for pi in self.train_fold]
            self.valid_indices = [self.pi_to_index[pi] for pi in self.valid_fold]

        self.input_size = self[0][0]["feature"].shape[0]

    def get_sequences_and_labels(self):
        sequences = {}
        labels = {}
        if self.test:
            info = "no_one_msa_info.txt"
        else:
            info = "info.txt"
        with open(os.path.join(self.dataset_dir, info)) as f:
            f.readline()
            line = f.readline()
            while line != "":
                line = line.strip().split("\t")
                key = line[0] + "_" + line[1] + "/" + line[2]
                sequences[key] = line[3]
                labels[key] = np.array([True if aa == "1" else False for aa in line[4]])
                line = f.readline()
        return sequences, labels

    def get_npy(self, name, flag=True):
        if not flag:
            return None
        mapping = {}
        print("Loading", name)
        tmp = glob(os.path.join(self.preprocessed_dir, "*", name + "_?.npy"))
        if tmp == []:
            tmp = glob(os.path.join(self.msa_dir, "*", name + "_?.npy"))
        if tmp == []:
            tmp = glob(os.path.join(self.raw_dir, "*", name + "_?.npy"))
        tmp = sorted(tmp)
        if self.hparams.progress_bar_refresh_rate != 0:
            tmp = tqdm(tmp)
        for file in tmp:
            pis, chain = file.split("/")[-2:]
            chain = chain[-5:-4]
            mapping[pis + "/" + chain] = np.load(file).astype(np.float32)
        return mapping

    def get_coords(self, pisc):
        pis, c = pisc.split("/")
        if self.hparams.use_cb_coords:
            return np.load(os.path.join(self.raw_dir, pis, "cb_coords_" + c + ".npy"))
        return np.load(os.path.join(self.raw_dir, pis, "ca_coords_" + c + ".npy"))

    def get_fold(self):
        with open(os.path.join(self.splits_dir, "train_ids_fold" + self.fold)) as f:
            train = sorted([line.strip() for line in f.readlines()])
        with open(os.path.join(self.splits_dir, "test_ids_fold" + self.fold)) as f:
            valid = sorted([line.strip() for line in f.readlines()])
        return train, valid

    def compute_pos_weight(self):
        zeros = 0
        ones = 0
        for pi in tqdm(self.train_fold, leave=False):
            pis = self.pi_to_pis[pi][0]
            for pisc in self.pis_to_pisc[pis]:
                try:
                    y = self.labels[pisc]
                except KeyError:
                    print(pi, pisc)
                one = np.count_nonzero(y)
                ones += one
                zeros += len(y) - one
        pos_weight = [zeros / ones]
        print(zeros, ones, "Done")
        return pos_weight

    def __getitem__(self, index):
        pi = self.dataset[index]
        # Taking the first structure available
        pis = self.pi_to_pis[pi][0]
        # For all available chains
        data = {}
        meta = {}
        for i, pisc in enumerate(self.pis_to_pisc[pis]):
            _data = {}
            _meta = {}

            mpisc = self.pisc_to_mpisc[pisc]
            sequence = self.sequences[pisc]
            _meta["pisc"] = [pisc]
            _meta["mpisc"] = [mpisc]
            _meta["sequence"] = [sequence]

            oh = np.array([np.eye(len(AMINO_ACIDS))[AA_DICT[aa]] for aa in sequence]).T
            pe = np.arange(1, len(sequence) + 1).reshape((1, -1)) / len(sequence)
            inputs = [oh, pe]
            if self.hparams.use_aap:
                inputs.append(self.aap[mpisc])
            if self.hparams.use_pssm:
                inputs.append(self.pssm[mpisc])
            if self.hparams.use_ss2:
                inputs.append(self.ss2[mpisc])
            if self.hparams.use_solv:
                inputs.append(self.solv[mpisc])

            if self.hparams.use_ss3:
                inputs.append(self.spot_ss3[mpisc] / 100.0)
            if self.hparams.use_ss8:
                inputs.append(self.spot_ss8[mpisc] / 100.0)
            if self.hparams.use_asa:
                # RSA = ASA / MaxASA
                tmp = self.spot_asa[mpisc]
                inputs.append(tmp / tmp.max())
            if self.hparams.use_cn13:
                # Normalize the Contact Number
                tmp = self.spot_cn13[mpisc]
                inputs.append(tmp / tmp.max())
            if self.hparams.use_hse:
                # Normalize the HSE
                tmp = self.spot_hse[mpisc]
                # Divide each row by the maximum
                inputs.append(tmp / tmp.max(axis=1)[:, None])
            if self.hparams.use_torsion:
                # Normalize or convert to sin and cos
                inputs.append(self.spot_torsion[mpisc] / 360.0)

            _data["feature"] = np.vstack(inputs).astype(np.float32)
            _data["label"] = self.labels[pisc].astype(np.float32)
            if self.test or index in self.valid_indices:
                _data["coords"] = self.get_coords(pisc).astype(np.float32)

            if i == 0:
                data = _data
                meta = _meta
            else:
                for key in _data:
                    data[key] = np.hstack((data[key], _data[key]))
                for key in _meta:
                    meta[key] += _meta[key]
        return data, meta

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    # A collate function to merge samples into a minibatch, will be used by DataLoader
    def collate_fn(samples):
        # samples is a list of tuples: len(samples) = Batch Size
        # Each tuple is of the form (data, meta)
        # Sort the samples in decreasing order of their length
        samples.sort(key=lambda sample: len(sample[0]["label"]), reverse=True)
        bs = len(samples)

        meta = {}
        meta["length"] = [0] * bs
        for i, (d, m) in enumerate(samples):
            meta["length"][i] = len(d["label"])
            if i == 0:
                for key in m:
                    meta[key] = []
            for key, val in m.items():
                meta[key] += [val]

        data = {}
        fdata = samples[0][0]
        for key in fdata.keys():
            data[key] = np.zeros((bs, *fdata[key].shape), dtype=fdata[key].dtype)
            for i, (d, _) in enumerate(samples):
                data[key][i, ..., : meta["length"][i]] = d[key][
                    ..., : meta["length"][i]
                ]
            data[key] = from_numpy(data[key])
        return data, meta

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--data-dir",
            default="../data/",
            type=str,
            help="Location of data directory. Default: %(default)s",
        )
        parser.add_argument(
            "--fold",
            metavar="NUMBER",
            type=int,
            default=0,
            help="Cross Validation fold number to train on. Default: %(default)d",
        )
        parser.add_argument(
            "--pos-weight",
            type=float,
            default=None,
            help="Provide a positional weight for the binding residue class. \
                If 0.0, it will be calculated from training set. \
                If None, it will be calculated on the fly for each batch. Default: %(default)s",
        )

        parser.add_argument(
            "--fixed-length",
            metavar="LENGTH",
            type=int,
            help="Use a fixed length input instead of variable lengths",
        )

        parser.add_argument(
            "--aap",
            dest="use_aap",
            action="store_true",
            help="Use amino acid probabilities in input features for the model. Default: %(default)s",
        )
        parser.add_argument("--no-aap", dest="use_aap", action="store_false")
        parser.set_defaults(use_aap=False)

        parser.add_argument(
            "--pssm",
            dest="use_pssm",
            action="store_true",
            help="Use Position Specific Scoring Matrix in input features for the model. Default: %(default)s",
        )
        parser.add_argument("--no-pssm", dest="use_pssm", action="store_false")
        parser.set_defaults(use_pssm=True)

        parser.add_argument(
            "--ss2",
            dest="use_ss2",
            action="store_true",
            help="Use secondary structure predicted using PSIPRED. Default: %(default)s",
        )
        parser.add_argument("--no-ss2", dest="use_ss2", action="store_false")
        parser.set_defaults(use_ss2=True)

        parser.add_argument(
            "--solv",
            dest="use_solv",
            action="store_true",
            help="Use solvent accessibilities predicted using SOLVPRED. Default: %(default)s",
        )
        parser.add_argument("--no-solv", dest="use_solv", action="store_false")
        parser.set_defaults(use_solv=True)

        parser.add_argument(
            "--ss3",
            dest="use_ss3",
            action="store_true",
            help="Use 3-state secondary structure predicted using SPOT-1D. Default: %(default)s",
        )
        parser.add_argument("--no-ss3", dest="use_ss3", action="store_false")
        parser.set_defaults(use_ss3=False)

        parser.add_argument(
            "--ss8",
            dest="use_ss8",
            action="store_true",
            help="Use 8-state secondary structure predicted using SPOT-1D. Default: %(default)s",
        )
        parser.add_argument("--no-ss8", dest="use_ss8", action="store_false")
        parser.set_defaults(use_ss8=False)

        parser.add_argument(
            "--asa",
            dest="use_asa",
            action="store_true",
            help="Use Accessible Surface area predicted using SPOT-1D. Default: %(default)s",
        )
        parser.add_argument("--no-asa", dest="use_asa", action="store_false")
        parser.set_defaults(use_asa=False)

        parser.add_argument(
            "--hse",
            dest="use_hse",
            action="store_true",
            help="Use Half Sphere Exposure predicted using SPOT-1D. Default: %(default)s",
        )
        parser.add_argument("--no-hse", dest="use_hse", action="store_false")
        parser.set_defaults(use_hse=False)

        parser.add_argument(
            "--torsion",
            dest="use_torsion",
            action="store_true",
            help="Use Torsion angles predicted using SPOT-1D. Default: %(default)s",
        )
        parser.add_argument("--no-torsion", dest="use_torsion", action="store_false")
        parser.set_defaults(use_torsion=False)

        parser.add_argument(
            "--cn13",
            dest="use_cn13",
            action="store_true",
            help="Use Contact Number predicted using SPOT-1D. Default: %(default)s",
        )
        parser.add_argument("--no-cn13", dest="use_cn13", action="store_false")
        parser.set_defaults(use_cn13=False)

        parser.add_argument(
            "--cb-coords",
            dest="use_cb_coords",
            action="store_true",
            help="Use C-Beta coordinates for validation and testing metrics instead of C-Alpha. Default: %(default)s",
        )
        parser.set_defaults(use_cb_coords=False)

        return parser
