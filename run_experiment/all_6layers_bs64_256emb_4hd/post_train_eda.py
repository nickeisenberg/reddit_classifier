import os
import matplotlib.pyplot as plt
import pandas as pd
from torch import load

loss_log_root = "./run_experiment/all_6layers_bs64_256emb_4hd/loss_logs"
train_loss_log = os.path.join(loss_log_root, "train_log.csv")
val_loss_log = os.path.join(loss_log_root, "validation_log.csv")
train = pd.read_csv(train_loss_log)
val = pd.read_csv(val_loss_log)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].plot(
    train["total_loss"].to_numpy().reshape((-1, 1091)).mean(axis=1),
    label="train loss"
)
ax[0].plot(
    val["total_loss"].to_numpy().reshape((-1, 297)).mean(axis=1),
    label="validation loss"
)
ax[1].plot(
    train["accuracy"].to_numpy().reshape((-1, 1091)).mean(axis=1),
    label="train accuracy (%)"
)
ax[1].plot(
    val["accuracy"].to_numpy().reshape((-1, 297)).mean(axis=1),
    label="validation accuracy (%)"
)
ax[0].legend()
ax[1].legend()
fig.supxlabel("epochs")
fig.savefig(os.path.join(loss_log_root, "loss_and_accuracy.png"))
plt.show()

sd = "run_experiment/all_6layers_bs64_256emb_4hd/state_dicts/validation_ckp.pth"
sd = load(sd, map_location="cpu")

sd["EPOCHS_RUN"]
