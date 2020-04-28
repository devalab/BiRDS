import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

from datasets import collate_fn, Kalasanty
from metrics import batch_loss, batch_metrics
from models import ResNet

torch.manual_seed(42)
device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using available GPU")
    device = torch.device("cuda")


# Define the main training loop
def train_loop(model, dl, pr=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dl, leave=False)
    for i, batch_el in enumerate(pbar):
        X, y, lengths = batch_el
        optimizer.zero_grad()
        y_pred = model(X, lengths)
        loss = batch_loss(y_pred, y, lengths, criterion=criterion)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % pr == pr - 1:
            pbar.set_postfix({"train_loss": loss.item()})
    print(f"Train --- %.8f" % (running_loss / len(dl)))


# Define the main validation loop
def valid_loop(model, dl):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dl, leave=False)
        for i, batch_el in enumerate(pbar):
            X, y, lengths = batch_el
            y_pred = model(X, lengths)
            metrics = batch_metrics(y_pred, y, lengths)
            metrics["loss"] = batch_loss(y_pred, y, lengths, criterion=criterion).item()
            pbar.set_postfix(metrics)
            if i == 0:
                running_metrics = metrics
                continue
            for key in metrics:
                running_metrics[key] += metrics[key]
        print("Validation --- ", end="")
        for key in metrics:
            print(f"%s: %.5f" % (key, (running_metrics[key] / len(dl))), end=" ")
        print()


max_epochs = 50
learning_rate = 0.02
dataset = Kalasanty(precompute_class_weights=True)
criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(dataset.pos_weight)).to(device)
feat_vec_len = dataset[0][0].shape[0]
models = []
optimizers = []

for i, (train_indices, valid_indices) in enumerate(dataset.custom_cv()):
    # model = StackedNN(feat_vec_len).to(device)
    model = ResNet(feat_vec_len, layers=[2, 2, 2, 2], kernel_sizes=[7, 7]).to(device)
    #     print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    models.append(model)
    optimizers.append(optimizer)
    print()
    print("Model #" + str(i + 1), "--------------------------------------------")
    for epoch in range(max_epochs):
        print("Epoch", str(epoch), end=" ")
        # Don't use multiprocessing here since our dataloading is I/O bound and not CPU
        train_dl = DataLoader(
            dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(train_indices),
            collate_fn=collate_fn,
        )
        train_loop(model, train_dl)
        valid_dl = DataLoader(
            dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(valid_indices),
            collate_fn=collate_fn,
        )
        valid_loop(model, valid_dl)
