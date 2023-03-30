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
            if not file.endswith(".jpg"): continue
            images_locations[file[:len(file)-4]].append(os.path.join(root, file))
    return images_locations

def get_unprocessed_image(images, width, height):
    """Get the unprocessed image from the list of images.

    Args:
        images (list): A list of images
        width (int): The width of the image
        height (int): The height of the image

    Returns:
        images (list): A list of images
    """
    raw_images = []
    for img_path in images:
        # check if the image is broken
        try:
            img = Image.open(img_path)
            img.verify()
        except UnidentifiedImageError as e:
            print(e)
            continue
        # ignore the image if the width and height do not match the CSV file
        if img.size[0] != width or img.size[1] != height:
            continue
        raw_images.append(img_path)
    return raw_images 

def validate_images(images, width, height):
    count = 0
    new_images = []
    for img_path in images:
        # check if the image is broken
        try:
            img = Image.open(img_path)
            img.verify()
        except UnidentifiedImageError as e:
            print(e)
            continue
        # ignore the image if the width and height do not match the CSV file
        if img.size[0] != width or img.size[1] != height:
            count += 1 
        new_images.append(img_path)
    return new_images, count 

def extract_data(station_csv_dir, dest_dir, images_locations, log_file, random_name):
    """Extract the raw data and store it in a more organized manner.

    Args:
        station_csv_dir (string): Path to the directory that contains the csv files  
        dest_dir (string): Path to the directory to store the extracted data  
        images_locations (defaultdict(list)): A dictionary that maps the id of the image to the location of the image (i.e., the key is the id of the image, and the value is the system path to the image) 
        log_file (string): Path to the log file
        random_name (bool): Use a random name for images 
    """
    notfound_count = 0 
    found_count = 0
    processed = 0
    start = time.time()

    # read in the csv files in the station_csv_dir
    for file in os.listdir(station_csv_dir):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(station_csv_dir, file))

        # iterate through the rows of the csv file, only using the id and classification_label columns
        for id, label, height, width in zip(list(df["Id"]), list(df["validatedClass"]), list(df["Image.Height"]), list(df["Image.Width"])):
            # check if the id or label is a float (i.e., NaN in this case)
            if isinstance(label, float) or isinstance(id, float):
                print("Found NaN in the csv file: ", file)
                continue 
            
            # check if the id specified in the csv file is in the raw dataset 
            if id not in images_locations:
                # write to a log file
                with open(log_file, "a") as f:
                    f.write(id + "\n")
                notfound_count += 1
                print("Not found: ", id)
                continue
            
            # get the correct image path
            if len(images_locations[id]) > 1:
                print("Found duplicate images: ", images_locations[id])
                image_path_s = get_unprocessed_image(images_locations[id], width, height)
            else:
                image_path_s = images_locations[id]
                image_path_s, val = validate_images(image_path_s, width, height)
                processed += val


            # ignore duplicates with identical resolution  
            if len(image_path_s) != 1: continue

            # move the images to the destination folder
            for img_path in image_path_s:
                # path of the current class folder
                class_dir = os.path.join(dest_dir, label)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                # copy the image to the destination folder
                if random_name:
                    shutil.copy(img_path, os.path.join(class_dir, uuid.uuid4().hex + ".jpg"))
                else:
                    shutil.copy(img_path, os.path.join(class_dir, id + ".jpg"))
                found_count += 1
                print("Transferred images: ", found_count)

    print(f"Finished organizing data, used approximately {time.time() - start} seconds.")
    print(f"Number of successfully transfers: {found_count}, inwhich {processed} were processed and {found_count-processed} not processed.")
    print(f"Found {notfound_count} images that were not found in the raw dataset.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Dataset Parser",
    description="Parse and convert the raw dataset into a suitable format for classification algorithms. The dataset can be split into test, train, and validation with a pre-defined ratio (0.8:0.1:0.1). If you wish to use another ratio, please use the ds_splitter.py script and set split to false.", 
    epilog="This program is a part of a bachelor thesis project.")
    parser.add_argument('-r', "--root", help="Path to the root directory of the raw dataset")
    parser.add_argument('-c', "--csv", help="Path to the directory that contains the csv files")
    parser.add_argument('-d', "--dest", help="Path to the directory to store the extracted data")
    parser.add_argument('-l', "--log", default="./unkown_ids.txt", help="Path to the log file")
    parser.add_argument("--split", type=bool,default=True, help="Split the dataset into test, train, and validation sets (default: True)")
    parser.add_argument("--random_name", type=bool, default=False, help="Use a random name for images (default: False)")
    args = parser.parse_args()
    images_locations = get_images_locations(args.root)
    if args.split:
        # create a temporary directory to store the extracted data
        tmp_dir = uuid.uuid4().hex
        extract_data(args.csv, tmp_dir, images_locations, args.log, args.random_name)
        # Don't provide the split argument for none default split arguments.
        # Configure the split arguments by executing the ds_splitter.py script directly. 
        split(tmp_dir, args.dest, 0.8, 0.1, 0.1, 1337)
        # remove the temporary directory
        shutil.rmtree(tmp_dir)
    else:
        extract_data(args.csv, args.dest, images_locations, args.log, args.random_name)