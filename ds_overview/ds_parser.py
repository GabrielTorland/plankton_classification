import os
import shutil
import pandas as pd
import argparse
from datetime import datetime
import time


def get_images_locations(root_dir):
    """Create a dictionary that stores the location of all the jpg files in the raw dataset.

    Args:
        root_dir (string): Path to the root directory of the raw dataset 

    Returns:
        dictionary/map: A dictionary that maps the id of the image to the location of the image (i.e., the key is the id of the image, and the value is the system path to the image) 
    """    
    images_locations = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                images_locations[file[:len(file)-4]] = os.path.join(root, file)
    return images_locations

def extract_data(station_csv_dir, dest_dir, images_locations):
    """Extract the raw data and store it in a more organized manner.

    Args:
        station_csv_dir (string): Path to the directory that contains the csv files  
        dest_dir (string): Path to the directory to store the extracted data  
        images_locations (dictionary): A dictionary that maps the id of the image to the location of the image (i.e., the key is the id of the image, and the value is the system path to the image) 
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
        for id, label, date in zip(list(df["Id"]), list(df["validatedClass"]), list(df["Date"])):
            # check if the id or label is a float (i.e., NaN in this case)
            if isinstance(label, float) or isinstance(id, float) or isinstance(date, float):
                continue
            # convert the date to a datetime object
            if "/" in date:
                date_obj = datetime.strptime(date, "%m/%d/%Y")
            elif "-" in date:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            else:
                raise NotImplementedError("Unknown date format")
            # Exclude the December images from all years except 2016 and 2018
            if date_obj.month == 12 and date_obj.year not in [2016, 2018]:
                continue
            # check if the id specified in the csv file is in the raw dataset 
            if id not in images_locations:
                # write to a log file
                with open("unkown_ids.txt", "a") as f:
                    f.write(id + "\n")
                notfound_count += 1
                continue
            # get the path to the image
            img_path = images_locations[id]
            tmp = os.path.join(dest_dir, label)
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            # copy the image to the destination folder
            shutil.copy(img_path, os.path.join(tmp, id + ".jpg"))
            found_count += 1
            print("Transferred images: ", found_count)

    print("Finished organizing data, used approximately ", time.time() - start, " seconds")
    print("Number of successfully transferred images: ", found_count)
    print("Number of unknown ids detected: ", notfound_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse the raw dataset to a more organized dataset for training a machine learning model")
    parser.add_argument("--root", help="Path to the root directory of the raw dataset")
    parser.add_argument("--csv", help="Path to the directory that contains the csv files")
    parser.add_argument("--dest", help="Path to the directory to store the extracted data")
    args = parser.parse_args()
    #images_locations = get_images_locations(args.root)

    #extract_data(args.csv, args.dest, images_locations)

    images_locations = get_images_locations("/home/olepedersen/Downloads/IHLS")
    extract_data("/home/olepedersen/Downloads/station_csv", "/home/olepedersen/source/repos/plankton_classification/baseline_training_set", images_locations)