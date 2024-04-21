import matplotlib.pyplot as plt
import pandas as pd

train_loss_log = "loss_logs/train_log.csv"
val_loss_log = "loss_logs/validation_log.csv"

train = pd.read_csv(train_loss_log)
val = pd.read_csv(val_loss_log)

fig, ax = plt.subplots(1, 2)
ax[0].plot(train["total_loss"])
ax[0].plot(val["total_loss"])
ax[1].plot(train["accuracy"])
ax[1].plot(val["accuracy"])
plt.show()
