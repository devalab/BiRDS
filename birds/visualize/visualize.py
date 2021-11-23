import os
from argparse import ArgumentParser
from shutil import copyfile

import numpy as np
from birds.datasets import Birds, scPDB
from birds.metrics import DCC, batch_work
from birds.test import load_nets_frozen, predict
from flask import Flask, render_template
from pytorch_lightning.utilities.apply_func import move_data_to_device
from tqdm.auto import tqdm


def get_pred_metrics(nets, dataloader):
    out = {}
    thresh = sum([net.hparams.threshold for net in nets]) / len(nets)
    print("Running through dataset")
    for _, batch in tqdm(enumerate(dataloader)):
        data, meta = scPDB.collate_fn([batch])
        data, meta = move_data_to_device((data, meta), nets[0].device)
        y_pred = predict(nets, data, meta)
        y_pred, y_true = batch_work(y_pred, data["label"], meta["length"])
        metric = nets[0].test_metrics(y_pred, y_true.int())
        metric = {
            k[2:]: np.nan_to_num(v.detach().cpu().numpy(), nan=-1, posinf=-1, neginf=-1).item()
            for k, v in metric.items()
        }
        # metric["ConfusionMatrix"] = nets[0].test_figure_metrics(y_pred, y_true.int())["t_ConfusionMatrix"]
        dcc = (
            move_data_to_device(DCC((y_pred >= thresh).bool(), y_true.bool(), data, meta), "cpu")[0]
            .detach()
            .cpu()
            .numpy()
        )
        metric["dcc"] = np.nan_to_num(dcc, nan=-1, posinf=-1, neginf=-1).item()
        out[meta["pisc"][0][0].split("/")[0]] = {
            "data": {k: v[0] for k, v in data.items()},
            "meta": {k: v[0] for k, v in meta.items()},
            "y_pred": (y_pred >= thresh).bool(),
            "metrics": metric,
        }
    return dict(sorted(out.items(), key=lambda item: item[1]["metrics"]["MatthewsCorrcoef"], reverse=True))


def parse_arguments():
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    parser.add_argument(
        "--ckpt_dir",
        default="../../model",
        type=str,
        help="Checkpoint directory containing checkpoints of all 10 folds. Default: %(default)s",
    )
    parser.add_argument(
        "--data-dir",
        default="../../data",
        type=str,
        help="Location of data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus to use for computation. Default: %(default)d",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="Run the models on their validation sets. Default: %(default)s",
    )
    parser.set_defaults(validate=False)
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(debug=False)
    hparams = parser.parse_args()
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    hparams.ckpt_dir = os.path.abspath(os.path.expanduser(hparams.ckpt_dir))
    return hparams


hparams = parse_arguments()
hparams.debug = False
validation = {}
test = {}
if not hparams.debug:
    nets = load_nets_frozen(hparams)
    nets[0].hparams.batch_size = 1
    datamodule = Birds(nets[0].hparams)
    hparams.device = nets[0].device
    if hparams.validate:
        validation = get_pred_metrics([nets[0]], datamodule.train_ds)
    else:
        test = get_pred_metrics(nets, datamodule.test_ds)

app = Flask(__name__, static_url_path="")


@app.route("/")
def main():
    if hparams.debug:
        return render_template("index.html", validation={}, test={})
    return render_template("index.html", validation=validation, test=test)


@app.route("/<pis>")
def show(pis):
    if pis == "favicon.ico":
        return ""
    if hparams.debug:
        return render_template(
            "show.html",
            protein="debug/protein.pdb",
            ligand="debug/ligand.mol2",
            selections="27:A,58:A;30:A,103:A;82:A,127:A",
            metrics={},
        )
    sv = pis + "/"
    if not os.path.exists("./static/" + sv):
        os.makedirs("./static/" + sv)
    if pis in test:
        pre = os.path.join(hparams.data_dir, "2018_scPDB", "raw", pis)
        hlp = test[pis]
    else:
        pre = os.path.join(hparams.data_dir, "scPDB", "raw", pis)
        hlp = validation[pis]
    protein = sv + "protein.pdb"
    ligand = sv + "ligand.mol2"
    copyfile(os.path.join(pre, "reindexed_protein.pdb"), "./static/" + protein)
    copyfile(os.path.join(pre, "ligand.mol2"), "./static/" + ligand)

    metrics = hlp["metrics"]
    y_pred = hlp["y_pred"]
    data = hlp["data"]
    meta = hlp["meta"]
    y_true = data["label"].bool()
    # y_pred = (torch.sigmoid(y_pred[0]) > 0.5).bool()
    # y_pred = (y_pred >= 0.5).bool()
    # print(y_true)
    # print(y_pred)
    selections = [""] * 3
    idx = 0
    for i, seq in enumerate(meta["sequence"]):
        ln = len(seq)
        pisc = meta["pisc"][i]
        chain = pisc[-1]
        for j in range(idx, idx + ln):
            if y_true[j] and y_pred[j]:
                hlp = 0
            elif y_true[j]:
                hlp = 1
            elif y_pred[j]:
                hlp = 2
            else:
                continue
            selection = str(j - idx + 1) + ":" + chain
            selections[hlp] += selection + ","
        idx += ln
    # print(final_selections)
    return render_template(
        "show.html",
        protein=protein,
        ligand=ligand,
        selections=";".join(selections),
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run()
