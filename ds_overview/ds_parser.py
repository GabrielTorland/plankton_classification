import os
import shutil
import pandas as pd
import argparse
import time
from collections import defaultdict
import uuid
from PIL import Image
from PIL import UnidentifiedImageError
from ds_splitter import split


def get_images_locations(root_dir):
    """Create a dictionary that stores the location of all the jpg files in the raw dataset.

    Args:
        root_dir (string): Path to the root directory of the raw dataset 

    Returns:
        defaultdeict(list) : A dictionary that maps the id of the image to the location of the image (i.e., the key is the id of the image, and the value is the system path to the image) 
    """    
    images_locations = defaultdict(list) 
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                images_locations[file[:len(file)-4]].append(os.path.join(root, file))
                if len(images_locations[file[:len(file)-4]]) > 1:
                    print("Duplicate image: ", file[:len(file)-4])
    return images_locations

def extract_data(station_csv_dir, dest_dir, images_locations, log_file):
    """Extract the raw data and store it in a more organized manner.

    Args:
        station_csv_dir (string): Path to the directory that contains the csv files  
        dest_dir (string): Path to the directory to store the extracted data  
        images_locations (defaultdict(list)): A dictionary that maps the id of the image to the location of the image (i.e., the key is the id of the image, and the value is the system path to the image) 
    """
    notfound_count = 0 
    found_count = 0
    start = time.time()
    
    # read in the csv files in the station_csv_dir
    for file in os.listdir(station_csv_dir):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(station_csv_dir, file))
        # iterate through the rows of the csv file, only using the id and classification_label columns
        for id, label in zip(list(df["Id"]), list(df["validatedClass"])):
            # check if the id or label is a float (i.e., NaN in this case)
            if isinstance(label, float) or isinstance(id, float):
                continue 
            
            # check if the id specified in the csv file is in the raw dataset 
            if id not in images_locations:
                # write to a log file
                with open(log_file, "a") as f:
                    f.write(id + "\n")
                notfound_count += 1
                continue

            # move the images to the destination folder
            for img_path in images_locations[id]:
                # check if the image is broken
                try:
                    im = Image.open(img_path)
                    im.verify()
                except UnidentifiedImageError:
                    continue
                tmp = os.path.join(dest_dir, label)
                if not os.path.exists(tmp):
                    os.makedirs(tmp)
                # copy the image to the destination folder
                shutil.copy(img_path, os.path.join(tmp, uuid.uuid4().hex + ".jpg"))
                found_count += 1
                print("Transferred images: ", found_count)

    print("Finished organizing data, used approximately ", time.time() - start, " seconds")
    print("Number of successfully transferred images: ", found_count)
    print(f"Found {notfound_count} images that were not found in the raw dataset.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Dataset Parser", description="Parse and convert the raw dataset into a suitable format for classification algorithms.", epilog="This program is a part of a bachelor thesis project.")
    parser.add_argument('-r', "--root", help="Path to the root directory of the raw dataset")
    parser.add_argument('-c', "--csv", help="Path to the directory that contains the csv files")
    parser.add_argument('-d', "--dest", help="Path to the directory to store the extracted data")
    parser.add_argument('-l', "--log", default="./unkown_ids.txt", help="Path to the log file")
    parser.add_argument("--split", type=bool,default=True, help="Split the dataset into test, train, and validation sets")
    args = parser.parse_args()
    images_locations = get_images_locations(args.root)
    if args.split:
        tmp_dir = uuid.uuid4().hex
        extract_data(args.csv, tmp_dir, images_locations, args.log)
        # Don't provide the split argument for none default split arguments.
        # Configure the split arguments by executing the ds_splitter.py script directly. 
        split(tmp_dir, args.dest, 0.8, 0.1, 0.1, 1337)
        # remove the temporary directory
        shutil.rmtree(tmp_dir)
    else:
        extract_data(args.csv, args.dest, images_locations, args.log)
