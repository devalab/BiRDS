# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict, defaultdict
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from pbsp.metrics import batch_loss, batch_metrics, weighted_bce_loss, make_figure

SMOOTH = 1e-6


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
        self.pssm = self.get_npy("pssm")
        self.torsion = self.get_npy("torsion")
        self.asa = self.get_npy("asa")
        self.ss3 = self.get_npy("ss3")
        self.re = self.get_entropy("re")
        self.jsd = self.get_entropy("jsd")

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

    def get_entropy(self, name):
        mapping = {}
        tmp = glob(os.path.join(self.msa_dir, "*/?." + name))
        for file in sorted(tmp):
            pis = file.split("/")[-2]
            chain = file[-len(name) - 2]
            k = pis + "/" + chain
            with open(file) as f:
                mapping[k] = np.array(
                    [float(el) for el in f.readline().strip().split()]
                ).reshape(1, -1)
            mapping[k][mapping[k] == -1000.0] = 0.0
        return mapping

    def get_coords(self, pisc):
        pis, c = pisc.split("/")
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

            rt = np.array([AA_DICT[aa] / 20 for aa in sequence]).reshape(1, -1)
            pe = np.arange(1, len(sequence) + 1).reshape((1, -1)) / len(sequence)
            if mpisc not in self.torsion:
                torsion = np.zeros((2, len(sequence)))
            else:
                torsion = self.torsion[mpisc][:2] / 360.0
            if mpisc not in self.asa:
                asa = np.zeros((1, len(sequence)))
            else:
                asa = self.asa[mpisc] / self.asa[mpisc].max()
            if mpisc not in self.ss3:
                ss3 = np.zeros((3, len(sequence)))
            else:
                ss3 = self.ss3[mpisc] / 100.0
            if mpisc not in self.re:
                re = np.zeros((1, len(sequence)))
            else:
                re = self.re[mpisc]
            if mpisc not in self.jsd:
                jsd = np.zeros((1, len(sequence)))
            else:
                jsd = self.jsd[mpisc]

            inputs = [
                self.pssm[mpisc][:20],
                torsion,
                asa,
                ss3,
                re,
                jsd,
                rt,
                pe,
            ]

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
            data[key] = torch.from_numpy(data[key])
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
            action="store_true",
            help="Compute the positional weight of the binding residue class. Default: %(default)s",
        )
        parser.add_argument("--no-pos-weight", action="store_false")
        parser.set_defaults(compute_pos_weight=True)
        return parser


class Block(nn.Module):
    def __init__(self, channel, filter_shape, block_type="residual"):
        super().__init__()
        self.block_type = block_type
        self.filter_shape = filter_shape
        self.padding = filter_shape[0] // 2
        self.layer_norm = nn.LayerNorm(channel, eps=0.001)
        self.Conv = nn.Conv2d(
            filter_shape[2], filter_shape[3], (filter_shape[0], filter_shape[1])
        )

    def forward(self, X):
        block_ln = X.transpose(1, 2).transpose(2, 3)
        block_ln = self.layer_norm(block_ln)
        block_glu = torch.split(block_ln, self.filter_shape[2], 3)
        block_glu = block_glu[0] * torch.sigmoid(block_glu[1])
        block_glu = block_glu.transpose(2, 3).transpose(1, 2)

        if self.block_type == "norm":
            return block_glu

        block_glu = F.pad(block_glu, (0, 0, self.padding, self.padding))

        block_conv = self.Conv(block_glu)
        if self.block_type == "plain":
            return block_conv

        block_output = X + block_conv
        return block_output


def bce_loss(y_pred, y_true, **kwargs):
    return F.binary_cross_entropy(y_pred, y_true, reduction="mean")


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_cpus = 10
        self.train_ds = scPDB(hparams)
        if hparams.run_tests:
            self.test_ds = scPDB(hparams, test=True)
        self.stage_depth = 2  # the number of BasicBlocks in a Stage (N)
        self.kernel_width = 5  # Height (k)
        self.amino_dim = 30
        self.std_in_channel = 256  # (c)
        self.std_out_channel = self.std_in_channel * 2
        self.std_filter_shape = [
            self.kernel_width,
            1,
            self.std_in_channel,
            self.std_out_channel,
        ]

        self.Conv0 = nn.Conv2d(1, self.std_out_channel, (3, self.amino_dim), bias=False)

        stage_1 = []
        for i in range(self.stage_depth):
            stage_1.append(
                Block(self.std_out_channel, self.std_filter_shape, "residual")
            )
        stage_1.append(Block(self.std_out_channel, self.std_filter_shape, "plain"))
        self.stage_1 = nn.Sequential(*stage_1)

        stage_2 = []
        for i in range(self.stage_depth):
            stage_2.append(
                Block(self.std_out_channel, self.std_filter_shape, "residual")
            )
        stage_2.append(Block(self.std_out_channel, self.std_filter_shape, "norm"))
        self.stage_2 = nn.Sequential(*stage_2)

        self.Conv1 = nn.Conv2d(
            self.std_in_channel, self.std_in_channel, (1, 1), bias=True
        )
        self.Conv2 = nn.Conv2d(self.std_in_channel, 1, (1, 1), bias=True)
        # self.loss_func = bce_loss
        self.loss_func = weighted_bce_loss

    def forward(self, X, lengths):
        batch_size = len(lengths)

        # Norm
        x = torch.reshape(X, [batch_size, 1, -1, self.amino_dim])

        # Trans
        conv0_input = F.pad(x, (0, 0, 1, 1))
        conv0 = self.Conv0(conv0_input)

        # Stage 1
        buffer_tensor = self.stage_1(conv0)

        # Stage 2
        buffer_tensor = self.stage_2(buffer_tensor)

        # Proj
        fc0 = self.Conv1(buffer_tensor)
        fc0_relu = F.relu(fc0)
        fc0_drop = F.dropout(fc0_relu, 0.25)

        fc1 = self.Conv2(fc0_drop)
        fc1_relu = F.relu(fc1)
        # fc1_softmax = F.softmax(fc1_relu.squeeze(3), dim=1)[:, 1, :]

        return fc1_relu.squeeze()

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.train_indices),
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.valid_indices),
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01, weight_decay=0.2)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=5, verbose=True
            ),
            "monitor": "v_mcc",
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, curr_epoch, batch_nb, optim, optim_idx, *args, **kwargs):
        if self.trainer.global_step < (20000 // self.hparams.batch_size):
            for pg in optim.param_groups:
                pg["lr"] = 0.00001 * (
                    0.95
                    ** (self.trainer.global_step // (5000 // self.hparams.batch_size))
                )

        if self.trainer.global_step >= (
            20000 // self.hparams.batch_size
        ) and self.trainer.global_step < (100000 // self.hparams.batch_size):
            for pg in optim.param_groups:
                pg["lr"] = 0.0001 * (
                    0.96
                    ** (self.trainer.global_step // (1000 // self.hparams.batch_size))
                )

        if self.trainer.global_step >= (100000 // self.hparams.batch_size):
            for pg in optim.param_groups:
                pg["lr"] = 0.0002 * (
                    0.98
                    ** (self.trainer.global_step // (2000 // self.hparams.batch_size))
                )

        optim.step()
        optim.zero_grad()

    def training_step(self, batch, batch_idx):
        data, meta = batch
        y_pred = self(data["feature"], meta["length"])
        loss = batch_loss(
            y_pred,
            data["label"],
            meta["length"],
            self.loss_func,
            pos_weight=self.train_ds.pos_weight,
        )
        return {"loss": loss, "log": {"loss": loss}}

    def val_test_step(self, batch, batch_idx, prefix):
        data, meta = batch
        y_pred = self(data["feature"], meta["length"])
        metrics = OrderedDict(
            {
                prefix
                + "loss": batch_loss(
                    y_pred,
                    data["label"],
                    meta["length"],
                    self.loss_func,
                    pos_weight=[1.0],
                )
            }
        )
        for key, val in batch_metrics(
            y_pred,
            data,
            meta,
            # logger=self.logger,
            # epoch=self.current_epoch,
        ).items():
            if key.startswith("f_"):
                metrics["f_" + prefix + key[2:]] = val
            else:
                metrics[prefix + key] = val
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "v_")

    def validation_epoch_end(self, outputs):
        avg_metrics = OrderedDict(
            {
                key: torch.stack([el[key] for el in outputs]).mean()
                for key in outputs[0]
                if not key.startswith("f_")
            }
        )
        figure_metrics = OrderedDict(
            {
                key[2:]: torch.stack(sum([el[key] for el in outputs], [])).cpu().numpy()
                for key in outputs[0]
                if key.startswith("f_")
            }
        )
        for key, val in figure_metrics.items():
            self.logger.experiment.add_figure(
                key, make_figure(key[2:], val), self.current_epoch
            )
        return {**avg_metrics, "progress_bar": avg_metrics, "log": avg_metrics}

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "t_")

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @staticmethod
    def add_class_specific_args(parser):
        return parser


class MyModelCheckpoint(ModelCheckpoint):
    def format_checkpoint_name(self, epoch, metrics, ver=None):
        if self.filename == "{epoch}":
            self.filename = "{epoch}-{v_mcc:.3f}-{v_acc:.3f}-{v_f1:.3f}"
        return super().format_checkpoint_name(epoch, metrics, ver)


def main(hparams):
    torch.manual_seed(hparams.seed)
    # Logging to HOME so that all experiments are available for viewing on Cluster
    save_dir = os.getenv("HOME")
    logger = TestTubeLogger(save_dir=save_dir, name="logs", create_git_tag=True)
    checkpoint_callback = MyModelCheckpoint(
        monitor="v_mcc", verbose=True, save_top_k=3, mode="max",
    )
    bs = hparams.batch_size
    if hparams.progress_bar_refresh_rate is None:
        hparams.progress_bar_refresh_rate = 64 // bs
    const_params = {
        "max_epochs": hparams.net_epochs,
        "row_log_interval": 64 // bs,
        "log_save_interval": 256 // bs,
        "gradient_clip_val": 0,
    }
    hparams = Namespace(**vars(hparams), **const_params)
    if not hparams.resume_from_checkpoint:
        accumulate_grad_batches = {5: max(1, 16 // bs), 10: 64 // bs}
    else:
        accumulate_grad_batches = 1
    print(hparams)
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=0.5,
        profiler=True,
        accumulate_grad_batches=accumulate_grad_batches,
        # track_grad_norm=2,
        # fast_dev_run=True,
        # overfit_pct=0.05,
    )

    if hparams.load_from_checkpoint is None:
        net = Net(hparams)
    else:
        net = Net.load_from_checkpoint(hparams.load_from_checkpoint)

    trainer.fit(net)

    if hparams.run_tests:
        trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)

    # Trainer Args
    trainer_group = parser.add_argument_group("Trainer")
    trainer_group.add_argument(
        "--seed", default=42, type=int, help="Training seed. Default: %(default)d"
    )
    trainer_group.add_argument(
        "--gpus", default=1, type=int, help="Default: %(default)d"
    )
    trainer_group.add_argument(
        "--batch-size",
        metavar="SIZE",
        default=8,
        type=int,
        help="Default: %(default)d",
    )
    trainer_group.add_argument(
        "--net-epochs",
        metavar="EPOCHS",
        default=50,
        type=int,
        help="Main Net epochs. Default: %(default)d",
    )
    trainer_group.add_argument(
        "--no-progress-bar",
        dest="progress_bar_refresh_rate",
        action="store_const",
        const=0,
    )
    trainer_group.add_argument(
        "--test",
        dest="run_tests",
        action="store_true",
        help="Run tests on model. Default: %(default)s",
    )
    trainer_group.add_argument("--no-test", dest="run_tests", action="store_false")
    trainer_group.set_defaults(run_tests=False)
    trainer_group.add_argument(
        "--load-from-checkpoint",
        metavar="PATH",
        default=None,
        type=str,
        help="Load model from file path provided",
    )
    trainer_group.add_argument(
        "--resume-from-checkpoint",
        metavar="PATH",
        default=None,
        type=str,
        help="Resume trainer from the specified checkpoint provided",
    )

    # Dataset Args
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group = scPDB.add_class_specific_args(dataset_group)

    # Lightning Module Args
    net_group = parser.add_argument_group("Net")
    net_group = Net.add_class_specific_args(net_group)

    # Parse as hyperparameters
    hparams = parser.parse_args()
    main(hparams)
