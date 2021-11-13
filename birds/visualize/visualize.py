import os
from argparse import ArgumentParser
from shutil import copyfile

import torch
from birds.metrics import batch_metrics, batch_work
from birds.test import load_nets_frozen
from flask import Flask, render_template
from tqdm.auto import tqdm


def get_pred_metrics(nets, dataset):
    out = {}
    try:
        indices = dataset.valid_indices
    except AttributeError:
        indices = range(len(dataset.dataset))
    print("Running through dataset")
    for ind in tqdm(indices):
        pi = dataset.dataset[ind]
        if not hparams.use_pis:
            pis = dataset.pi_to_pis[pi][0]
        else:
            pis = pi
        data, meta = dataset[ind]
        meta["length"] = torch.tensor([len(data["label"])], device=hparams.device)
        for key in data:
            data[key] = torch.tensor([data[key]], device=hparams.device)
        y_preds = []
        for i, net in enumerate(nets):
            y_pred, y_true = batch_work(
                net(data["feature"], meta["length"]), data["label"], meta["length"]
            )
            y_preds.append((torch.sigmoid(y_pred) >= net.hparams.threshold).float())
        y_pred = torch.mean(torch.stack(y_preds), dim=0)
        metrics = batch_metrics(y_pred, data, meta, is_logits=False, threshold=0.499)
        metrics["f_dcc"] = metrics["f_dcc"][0]
        metrics = {key: val.item() for key, val in metrics.items() if key != "f_cm"}
        metrics["mcc"] *= 100.0
        metrics["acc"] *= 100.0
        metrics["iou"] *= 100.0
        metrics["precision"] *= 100.0
        metrics["recall"] *= 100.0
        metrics["f1"] *= 100.0
        out[pis] = {"data": data, "meta": meta, "y_pred": y_pred, "metrics": metrics}
        # break
    out = {
        k: v
        for k, v in sorted(
            out.items(), key=lambda item: item[1]["metrics"]["mcc"], reverse=True
        )
    }
    return out


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
    hparams.device = nets[0].device
    if hparams.validate:
        validation = get_pred_metrics([nets[0]], nets[0].train_ds)
    else:
        test = get_pred_metrics(nets, nets[0].test_ds)

app = Flask(__name__, static_url_path="")


@app.route("/")
def main():
    if hparams.debug:
        return render_template("index.html", validation={}, test={})
    return render_template("index.html", validation=validation, test=test)


@app.route("/<pis>")
def show(pis):
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
    y_true = data["label"][0].bool()
    # y_pred = (torch.sigmoid(y_pred[0]) > 0.5).bool()
    y_pred = (y_pred >= 0.5).bool()
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
