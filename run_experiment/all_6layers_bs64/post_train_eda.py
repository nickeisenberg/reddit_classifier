import matplotlib.pyplot as plt
import pandas as pd


train_loss_log = "loss_logs/train_log.csv"
val_loss_log = "loss_logs/validation_log.csv"

train = pd.read_csv(train_loss_log)
val = pd.read_csv(val_loss_log)

fig, ax = plt.subplots(1, 2)
ax[0].plot(
    train["total_loss"].to_numpy().reshape((-1, 1091)).mean(axis=1),
    label="train loss"
)
ax[0].plot(
    val["total_loss"].to_numpy().reshape((-1, 149)).mean(axis=1),
    label="validation loss"
)
ax[1].plot(
    train["accuracy"].to_numpy().reshape((-1, 1091)).mean(axis=1),
    label="train accuracy (%)"
)
ax[1].plot(
    val["accuracy"].to_numpy().reshape((-1, 149)).mean(axis=1),
    label="validation accuracy (%)"
)
ax[0].legend()
ax[1].legend()
fig.supxlabel("epochs")
plt.show()
