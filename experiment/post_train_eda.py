import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import load

loss_log_root = "./experiment/loss_logs"
train_loss_log = os.path.join(loss_log_root, "train_log.csv")
val_loss_log = os.path.join(loss_log_root, "validation_log.csv")
train = pd.read_csv(train_loss_log)
val = pd.read_csv(val_loss_log)


train_loss = train["total_loss"].to_numpy().reshape((-1, 1091)).mean(axis=1)
train_accuracy = train["accuracy"].to_numpy().reshape((-1, 1091)).mean(axis=1)
val_loss = val["total_loss"].to_numpy().reshape((-1, 297)).mean(axis=1)
val_accuracy = val["accuracy"].to_numpy().reshape((-1, 297)).mean(axis=1)


fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].plot(
    train_loss,
    label="train loss"
)
ax[0].plot(
    val_loss,
    label="validation loss"
)
ax[1].plot(
    train_accuracy,
    label="train accuracy (%)"
)
ax[1].plot(
    val_accuracy,
    label="validation accuracy (%)"
)
ax[0].legend()
ax[1].legend()
fig.supxlabel("epochs")
fig.savefig(os.path.join(loss_log_root, "loss_and_accuracy.png"))
plt.show()

sd = "experiment/state_dicts/validation_ckp.pth"
sd = load(sd, map_location="cpu")
