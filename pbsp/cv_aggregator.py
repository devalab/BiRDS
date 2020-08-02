#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
# from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pbsp.net import Net
from pytorch_lightning.utilities.apply_func import move_data_to_device
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def validate(net):
    dcc = []
    cm = defaultdict(int)
    fnl_metrics = {}
    for idx, batch in tqdm(enumerate(net.val_dataloader())):
        batch = move_data_to_device(batch, net.device)
        metrics = net.validation_step(batch, idx)
        dcc = dcc + [el.item() for el in metrics["f_v_dcc"]]
        cm["tn"] += metrics["f_v_cm"][0][0]
        cm["fp"] += metrics["f_v_cm"][0][1]
        cm["fn"] += metrics["f_v_cm"][0][2]
        cm["tp"] += metrics["f_v_cm"][0][3]
    fnl_metrics["cm"] = cm
    fnl_metrics["dcc"] = dcc
    return fnl_metrics


def get_best_ckpt(folder):
    tmp = [el[:-5].split("-") for el in sorted(os.listdir(folder))]
    tmp = sorted(
        tmp,
        key=lambda x: (
            float(x[1].split("=")[1]),
            float(x[2].split("=")[1]),
            float(x[3].split("=")[1]),
        ),
        reverse=True,
    )
    return os.path.join(folder, "-".join(tmp[0]) + ".ckpt")


def dcc_figure(values):
    values = np.nan_to_num(values, nan=20000, posinf=20000, neginf=20000)
    figure = plt.figure(figsize=(12, 12))
    x = np.linspace(0, 20, 1000)
    y = []
    for value in values:
        y.append(np.array([(value <= el).sum() / len(value) * 100 for el in x]))
    colours = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    for i, colour in enumerate(colours):
        y_new = gaussian_filter1d(y[i], sigma=5)
        plt.plot(x, y_new, colour, label="Fold " + str(i + 1))
    plt.legend()
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 101, 5))
    plt.title("Distance to the center of the binding site")
    plt.ylabel("Success Rate")
    plt.xlabel("Distance to binding site")
    plt.show()


#     return figure


# In[39]:


# parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
# parser.add_argument(
#     "ckpt_dir",
#     type=str,
#     help="Checkpoint directory containing checkpoints of all 10 folds",
# )
# parser.add_argument(
#     "--data-dir",
#     default="../../data",
#     type=str,
#     help="Location of data directory. Default: %(default)s",
# )
# args = parser.parse_args()
data_dir = os.path.abspath(os.path.expanduser("../../data"))
ckpt_dir = os.path.abspath(os.path.expanduser("~/logs/cv_0"))


# In[40]:


nets = []
metrics = []
for i in range(10):
    ckpt = get_best_ckpt(os.path.join(ckpt_dir, "fold_" + str(i), "checkpoints"))
    nets.append(Net.load_from_checkpoint(ckpt).cuda())
    nets[i].freeze()
    nets[i].eval()
    metrics.append(validate(nets[i]))


# In[41]:


dcc_figure([metric["dcc"] for metric in metrics])


# In[42]:


cm = defaultdict(int)
for metric in metrics:
    for key, val in metric["cm"].items():
        cm[key] += val


# In[44]:


from pbsp.metrics import *

print("IOU: ", IOU(cm).item())
print("MCC: ", MCC(cm).item())
print("ACCURACY: ", ACCURACY(cm).item())
print("RECALL: ", RECALL(cm).item())
print("PRECISION: ", PRECISION(cm).item())
print("F1: ", F1(cm).item())


# In[ ]:


# test_dl = nets[0].test_dataloader()
# device = nets[0].device
# for idx, batch in enumerate(test_dl):
#     data, meta = move_data_to_device(batch, device)
#     for i, net in enumerate(nets):
#         y_pred = net(data["feature"], meta["length"])