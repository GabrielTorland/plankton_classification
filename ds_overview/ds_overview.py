import seaborn as sns
import cv2
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))


"""Extract the data set from the given paths and return it as a dictionary"""
def dataset2dict(paths):
    """
    Opens a local data set and store it in a dictionary.

    Args:
        paths (list): list of the paths to the data set(i.e., training, test, validation). 

    Returns:
        dict(defaultdict(list)): The first key specifies the set (i.e., training, test, validation). The second key specifies the species. The value is a list of images.
    """    
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
    """
    Plots the distribution of the data set. One plot for each set and one plot for all sets together.

    Args:
        ds (dict(defaultdict(list))): The first key specifies the set (i.e., training, test, validation). The second key specifies the species. The value is a list of images.
    """    
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

"""Plot a pie chart of the data set"""
def plot_pie_chart(ds):
    """
    Plots a pie chart of the data set.

    Args:
        ds (dict(defaultdict(list))): The first key specifies the set (i.e., training, test, validation). The second key specifies the species. The value is a list of images.
    """    
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    
    labels = []
    sizes = []
    for (key, val) in ds["train"].items():
        labels.append(key)
        sizes.append(len(val))
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    wedges, autotexts = ax.pie(sizes, labels=labels, colors=colors, textprops=dict(color="w"))
    ax.legend(wedges, labels, title='Species', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Dataset Distribution")
    plt.show()    
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    #plt.axis('equal')
    #plt.show()

if __name__ == "__main__":
    paths = ["../baseline_training_set/train/", "../baseline_training_set/test/", "../baseline_training_set/validation/"] 
    ds = dataset2dict(paths)
    # plot_ds_dist(ds)
    plot_pie_chart(ds)
