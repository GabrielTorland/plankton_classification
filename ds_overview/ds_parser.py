import os
import shutil
import pandas as pd
import argparse


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
    # read in the csv files in the station_csv_dir
    for file in os.listdir(station_csv_dir):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(station_csv_dir, file))
        # iterate through the rows of the csv file, only using the id and classification_label columns
        for id, label in zip(df["id"], df["classification_label"]):
            # check if the id specified in the csv file is in the raw dataset 
            if id not in images_locations:
                raise(f"Image with id {id} not found")
            # get the path to the image
            img_path = images_locations[id]
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            # copy the image to the destination folder
            shutil.copy(img_path, dest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse the raw dataset to a more organized dataset for training a machine learning model")
    parser.add_argument("--root", help="Path to the root directory of the raw dataset")
    parser.add_argument("--csv", help="Path to the directory that contains the csv files")
    parser.add_argument("--dest", help="Path to the directory to store the extracted data")
    args = parser.parse_args()
    images_locations = get_images_locations(args.root)
    extract_data(args.csv, args.dest, images_locations)
    