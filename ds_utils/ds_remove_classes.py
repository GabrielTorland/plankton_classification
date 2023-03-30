import os
import argparse
import shutil

# change working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def remove_classes(source, whitelist):
    # remove classes from the training set
    for root, dirs, files in os.walk(os.path.join(source, "train")):
        for dir in dirs:
            if dir in whitelist: continue
            shutil.move(os.path.join(root, dir), os.path.join("backup", "train", dir))
    # remove classes from the validation set
    for root, dirs, files in os.walk(os.path.join(source, "val")):
        for dir in dirs:
            if dir in whitelist: continue
            shutil.move(os.path.join(root, dir), os.path.join("backup", "val", dir))
    # remove classes from the test set
    for root, dirs, files in os.walk(os.path.join(source, "test")):
        for dir in dirs:
            if dir in whitelist: continue
            shutil.move(os.path.join(root, dir), os.path.join("backup", "test", dir))
    print("Successfully applied whitelist filter on dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove classes from a dataset')
    parser.add_argument("-s", "--source", default="dataset", help="Path to the dataset")
    args = parser.parse_args()
    # manually specify the classes to include in the dataset
    whitelist = ["appendicularia", "bivalve", "ceratium", "ciliate", "copepod", "diatom", "dinoflagellate", "exuvie"] 
    remove_classes(args.source, whitelist)