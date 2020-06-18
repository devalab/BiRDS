import os
from argparse import ArgumentParser
from shutil import copyfile

import torch
from flask import Flask, render_template

from net import Net

data_dir = os.path.abspath("./data/")
parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
parser.add_argument("cpkt", type=str, help="Checkpoint file for loading model")
args = parser.parse_args()
net = Net.load_from_checkpoint(args.cpkt)
device = net.device
net.freeze()
net.eval()

app = Flask(__name__, static_url_path="")


@app.route("/")
def main():
    return render_template(
        "main.html",
        selections="select 18,19,20; color green; select 21,23,39; color red; select 40,41,60,61,78,304,305,306; color blue; select 18,19,20,21,23,39,40,41,60,61,78,304,305,306;",
        colors=[
            " select 18 and *.ca; label %n%R; color label magenta;select 19 and *.ca; label %n%R; color label magenta;select 20 and *.ca; label %n%R; color label magenta;",
            " select 21 and *.ca; label %n%R; color label magenta;select 23 and *.ca; label %n%R; color label magenta;select 39 and *.ca; label %n%R; color label magenta;",
            " select 40 and *.ca; label %n%R; color label magenta;select 41 and *.ca; label %n%R; color label magenta;select 60 and *.ca; label %n%R; color label magenta;select 61 and *.ca; label %n%R; color label magenta;select 78 and *.ca; label %n%R; color label magenta;select 304 and *.ca; label %n%R; color label magenta;select 305 and *.ca; label %n%R; color label magenta;select 306 and *.ca; label %n%R; color label magenta;  label %n%R; color label magenta;",
        ],
    )


@app.route("/<pis>")
def show(pis):
    pre = os.path.join(data_dir, "scPDB", "raw", pis)
    sv = "data/" + pis + "/"
    if not os.path.exists("./static/" + sv):
        os.makedirs("./static/" + sv)
    if not os.path.exists(pre):
        pre = os.path.join(data_dir, "2018_scPDB", "raw", pis)
        dataset = net.test_ds
    else:
        dataset = net.train_ds
    protein = sv + "protein.pdb"
    ligand = sv + "ligand.mol2"
    copyfile(os.path.join(pre, "reindexed_protein.pdb"), "./static/" + protein)
    copyfile(os.path.join(pre, "ligand.mol2"), "./static/" + ligand)
    idx = dataset.pi_to_index[pis[:4]]
    data, meta = dataset[idx]
    y_true = torch.tensor(data["label"], device=device).bool()
    y_pred = net(
        torch.tensor([data["feature"]], device=device),
        torch.tensor([len(data["label"])], device=device),
    )[0]
    y_pred = (torch.sigmoid(y_pred) > 0.5).bool()
    # print(y_true)
    # print(y_pred)
    selections = ["select "] * 3
    colors = ["color green; ", "color blue; ", "color red; "]
    final_selections = "select "
    labels = ""
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
            final_selections += selection + ","
            selections[hlp] += selection + ","
            labels += (
                "select " + selection + " and *.ca; label %n%R; color label magenta; "
            )
        idx += ln
    tmp = ""
    for i in range(3):
        if selections[i] != "select ":
            tmp += selections[i][:-1] + "; " + colors[i]
    final_selections = tmp + final_selections[:-1] + ";"
    # print(final_selections)
    return render_template(
        "main.html", protein=protein, ligand=ligand, selections=final_selections
    )
    return ""


if __name__ == "__main__":
    app.run()
