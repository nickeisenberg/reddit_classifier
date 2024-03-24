import numpy as np
import os

counts = []
for sub in os.listdir("data"):
    for root, _, files in os.walk(os.path.join("data", sub, "train")):
        for file in files:
            if not file.endswith(".txt"):
                continue
            with open(os.path.join(root, file), "r") as af:
                text = af.readline()
                counts.append(len(text.split()))

print(np.mean(counts))

print(np.max(counts))

print(np.min(counts))

print(np.quantile(counts, [.5, .6, .7, .8, .95]))
