import os
from argparse import ArgumentParser
from shutil import copyfile

import torch
from flask import Flask, render_template
from tqdm.auto import tqdm


def get_pred_metrics(dataset):
    out = {}
    try:
        indices = dataset.valid_indices
    except AttributeError:
        indices = range(len(dataset.dataset))
    print("Running through dataset")
    for ind in tqdm(indices):
        pi = dataset.dataset[ind]
        pis = dataset.pi_to_pis[pi][0]
        data, meta = dataset[ind]
        meta["length"] = torch.tensor([len(data["label"])], device=device)
        for key in data:
            data[key] = torch.tensor([data[key]], device=device)
        y_pred = net(data["feature"], meta["length"])
        metrics = batch_metrics(y_pred, data, meta)
        metrics["f_dcc"] = metrics["f_dcc"][0]
        metrics = {key: val.item() for key, val in metrics.items() if key != "f_cm"}
        metrics["mcc"] *= 100.0
        metrics["acc"] *= 100.0
        metrics["iou"] *= 100.0
        metrics["precision"] *= 100.0
        metrics["recall"] *= 100.0
        metrics["f1"] *= 100.0
        out[pis] = {"data": data, "meta": meta, "y_pred": y_pred, "metrics": metrics}
    out = {
        k: v
        for k, v in sorted(
            out.items(), key=lambda item: item[1]["metrics"]["mcc"], reverse=True
        )
    }
    return out


parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
parser.add_argument("cpkt", type=str, help="Checkpoint file for loading model")
parser.add_argument(
    "--data-dir",
    default="../../data",
    type=str,
    help="Location of data directory. Default: %(default)s",
)
parser.add_argument("--debug", action="store_true")
parser.set_defaults(debug=False)
args = parser.parse_args()
data_dir = os.path.abspath(args.data_dir)
if not args.debug:
    from pbsp.metrics import batch_metrics
    from pbsp.net import Net

    net = Net.load_from_checkpoint(args.cpkt).cuda()
    device = net.device
    print(device)
    net.freeze()
    net.eval()
    validation = get_pred_metrics(net.train_ds)
    test = get_pred_metrics(net.test_ds)

app = Flask(__name__, static_url_path="")


@app.route("/")
def main():
    if args.debug:
        return render_template("index.html", validation={}, test={})
    return render_template("index.html", validation=validation, test=test)


@app.route("/<pis>")
def show(pis):
    if args.debug:
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
        pre = os.path.join(data_dir, "2018_scPDB", "raw", pis)
        hlp = test[pis]
    else:
        pre = os.path.join(data_dir, "scPDB", "raw", pis)
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
    y_pred = (torch.sigmoid(y_pred[0]) > 0.5).bool()
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
