import argparse
import os
import shutil
import platform

def undo_split(source, destination):
    """This function takes the dataset and merges it into a single directory with subdirectories for each category.

    Args:
        source (str): Path to the dataset.
        destination (str): Path to the folder where the images will be stored.

    Raises:
        NotImplementedError: If the OS isn't supported. 
    """ 

    # create the destination directory if it does not exist
    if os.path.exists(destination):
        os.makedirs(destination)
    for root, dirs, files in os.walk(source):
        for file in files:
            if not file.endswith(".jpg"): continue
            # determine the category of the image
            if platform.system() == "Windows":
                category = root.split("\\")[-1]
            elif platform.system() == "Linux":
                category = root.split("/")[-1]
            else:
                raise NotImplementedError("This platform is not supported yet.")
            # create the category directory if it does not exist in the destination directory
            if not os.path.exists(os.path.join(destination, category)):
                os.makedirs(os.path.join(destination, category))
            # copy the file to the destination directory 
            shutil.copy(os.path.join(root, file), os.path.join(destination, category, file))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script extracts the images from the dataset.")
    parser.add_argument("-s", "--source", type=str, default="dataset", help="Path to the dataset (default=./dataset).")
    parser.add_argument("-d", "--destination", type=str, default="new_dataset", help="Path to the folder where the images will be stored (defult=./new_dataset).")
    args = parser.parse_args()
    undo_split(args.source, args.destination)