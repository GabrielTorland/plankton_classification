import seaborn as sns
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))



def dataset2dict(path):
    """Opens a local data set and store it in a dictionary.

    Args:
        path (string): The path to the data set. 

    Returns:
        dict(defaultdict(list)): The first key specifies the set (i.e., training, test, validation). The second key specifies the species. The value is the number of images.
    """    
    # data set in dictionary format
    ds = {"train": defaultdict(list), "test": defaultdict(list), "val": defaultdict(list)}   
    for folder in os.listdir(path):
        for category in os.listdir(path + folder):
            ds[folder][category.replace('_', ' ')] = len(os.listdir(path + folder + "/" + category))
    return ds


def plot_ds_dist(ds):
    """Plots the distribution of the data set. One plot for each set and one plot for all sets together.

    Args:
        ds (dict(defaultdict(list))): The first key specifies the set (i.e., training, test, validation). The second key specifies the species. The value is the number of images.
    """    
    df_train = pd.DataFrame()
    for (key, val) in ds["train"].items():
        df_train = df_train.append({"species": key, "frequency": val}, ignore_index=True)
    df_test = pd.DataFrame()
    for (key, val) in ds["test"].items():
        df_test = df_test.append({"species": key, "frequency": val}, ignore_index=True)
    df_val = pd.DataFrame()
    for (key, val) in ds["val"].items():
        df_val = df_val.append({"species": key, "frequency": val}, ignore_index=True)
    df_all = pd.DataFrame(columns=["set", "species", "frequency"])
    for item0, item1, item2 in zip(ds["test"].items(), ds["train"].items(), ds["val"].items()):
        df_all.loc[len(df_all)] = ["test", item0[0], item0[1]]
        df_all.loc[len(df_all)] = ["train", item1[0], item1[1]]
        df_all.loc[len(df_all)] = ["val", item2[0], item2[1]]
    h = sns.barplot(x="species", y="frequency", data=df_train, ci=0)
    plt.xticks(rotation=90)
    h.set_title("Training Set Distribution")
    #h.bar_label(h.containers[0])
    plt.subplots_adjust(right=1.4, left=0)
    plt.savefig("dist_train.png", bbox_inches='tight')
    h = sns.barplot(x="species", y="frequency", data=df_test, ci=0)
    plt.xticks(rotation=90)
    h.set_title("Test Set Distribution")
    #h.bar_label(h.containers[0])
    plt.subplots_adjust(right=1.4, left=0)
    plt.savefig("dist_test.png", bbox_inches='tight')
    h = sns.barplot(x="species", y="frequency", data=df_val, ci=0)
    plt.xticks(rotation=90)
    h.set_title("Validation Set Distribution")
    #h.bar_label(h.containers[0])
    plt.subplots_adjust(right=1.4, left=0)
    plt.savefig("dist_val.png", bbox_inches='tight')
    h = sns.barplot(x="species", y="frequency", data=df_all, ci=0, hue="set")
    plt.xticks(rotation=90)
    h.set_title("Data Set Distribution")
    #h.bar_label(h.containers[0])
    plt.subplots_adjust(right=1.4, left=0)
    plt.savefig("dist_pole.png", bbox_inches='tight')

    # write the exact percentage of each class in the data t
    for (key, val) in ds["train"].items():
        with open("ds_percentage.txt", "a") as f:
            # the distribution is the same for all sets
            # that's why we can only use the training set
            f.write(key + ": " + str(val / sum(ds["train"].values()) * 100) + '\n')


def plot_pie_chart(ds):
    """Plots a pie chart of the data set.

    Args:
        ds (dict(defaultdict(list))): The first key specifies the set (i.e., training, test, validation). The second key specifies the species. The value is a list of images.
    """    
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    
    labels = []
    sizes = []
    for (key, val) in ds["train"].items():
        labels.append(key)
        sizes.append(val)
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    wedges, autotexts = ax.pie(sizes, labels=labels, colors=colors, textprops=dict(color="w"))
    ax.legend(wedges, labels, title='Species', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Dataset Distribution")
    plt.savefig("dist_pie_chart.png", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots the distribution of the data set.")
    parser.add_argument("-s", "--source", type=str, help="The path to the data set.")
    args = parser.parse_args()
    ds = dataset2dict(args.source if args.source[-1] == "/" else args.source + "/")
    plot_ds_dist(ds)
    plot_pie_chart(ds)
