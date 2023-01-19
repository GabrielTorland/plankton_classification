import seaborn as sns
import cv2
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))


"""Extract the data set from the given paths and return it as a dictionary"""
def dataset2dict(paths):
    # tranlation table
    trans = {1: "train", 2: "test", 3: "validation"}
    # data set in dictionary format
    ds = {"train": defaultdict(list), "test": defaultdict(list), "validation": defaultdict(list)}   
    for i, path in enumerate(paths):
        for folder in os.listdir(path):
            for file in os.listdir(path + folder):
                if file.endswith(".jpg"):
                    ds[trans[i+1]][folder].append(cv2.imread(path + folder + "/" + file))
    return ds

"""Plot the distribution of the data set"""
def plot_ds_dist(ds):
    df_train = pd.DataFrame()
    for (key, val) in ds["train"].items():
        df_train = df_train.append({"species": key, "frequency": len(val)}, ignore_index=True)
    df_test = pd.DataFrame()
    for (key, val) in ds["test"].items():
        df_test = df_test.append({"species": key, "frequency": len(val)}, ignore_index=True)
    df_val = pd.DataFrame()
    for (key, val) in ds["validation"].items():
        df_val = df_val.append({"species": key, "frequency": len(val)}, ignore_index=True)
    df_all = pd.DataFrame(columns=["set", "species", "frequency"])
    for item0, item1, item2 in zip(ds["test"].items(), ds["train"].items(), ds["validation"].items()):
        df_all.loc[len(df_all)] = ["test", item0[0], len(item0[1])]
        df_all.loc[len(df_all)] = ["train", item1[0], len(item1[1])]
        df_all.loc[len(df_all)] = ["validation", item2[0], len(item2[1])]
    h = sns.barplot(x="frequency", y="species", data=df_train, ci=0)
    h.set_title("Training Set Distribution")
    #h.bar_label(h.containers[0])
    plt.show()
    h = sns.barplot(x="frequency", y="species", data=df_test, ci=0)
    h.set_title("Test Set Distribution")
    #h.bar_label(h.containers[0])
    plt.show()
    h = sns.barplot(x="frequency", y="species", data=df_val, ci=0)
    h.set_title("Validation Set Distribution")
    #h.bar_label(h.containers[0])
    plt.show()
    h = sns.barplot(x="species", y="frequency", data=df_all, ci=0, hue="set")
    h.set_title("Data Set Distribution")
    #h.bar_label(h.containers[0])
    plt.show()


if __name__ == "__main__":
    paths = ["../baseline_training_set/train/", "../baseline_training_set/test/", "../baseline_training_set/validation/"] 
    ds = dataset2dict(paths)
    plot_ds_dist(ds)