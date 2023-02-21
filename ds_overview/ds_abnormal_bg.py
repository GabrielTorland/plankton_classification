import cv2
import numpy as np
import os
import json
import shutil
import platform
import argparse


# change the current working directory to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def store_dist(jsonfile, avg_bgs, paths):
    """ Store the average background distribution and associated paths in a json file to reduce computation time in the future

    Args:
        jsonfile (str): Path to the json file 
        avg_bgs (list): List of average background pixel values
        paths (list): List of image paths
    """    
    json.dump({"avg_bgs" : avg_bgs, "paths": paths}, open(jsonfile, "w"))


def load_dist(jsonfile):
    """Load the average background distribution and associated paths from a json file

    Args:
        jsonfile (str): Path to the json file 

    Returns:
        list, list: List of average background pixel values, list of image paths
    """    
    data = json.load(open(jsonfile, "r"))
    return data["avg_bgs"], data["paths"]


def find_bg_pixels(gray):
    """Apply Gaussian blur then Otsu's binary thresholding to find the background pixels

    Args:
        gray (numpy.ndarray): Grayscaled image 

    Returns:
        numpy.ndarray: Background pixels 
    """    
    # apply Gaussian blur then Otsu's binary thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # determine the background pixels
    bg_pixels = np.where(thresh == 0)
    return bg_pixels


def get_background_dist(root_dir):
    """ Calculate the average background pixel value for each image in the dataset.
        The image is grayscaled before calculating the average background pixel value.

    Args:
        root_dir (str): Path to the dataset 

    Returns:
        list, list: List of average background pixel values, list of image paths 
    """    
    avg_bgs = []
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # open the image with opencv
            img = cv2.imread(os.path.join(root, file))
            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the background pixels
            bg_pixels = find_bg_pixels(gray) 
            # calculate the average background pixel value
            avg_bg = np.mean(gray[bg_pixels])
            avg_bgs.append(avg_bg)
            paths.append(os.path.join(root, file))
    return avg_bgs, paths


def manually_validate_anomalies(avg_bgs, paths, std_factor=4):
    """This function provides a way manually validate the background distribution by visualization.

    Args:
        avg_bgs (list): List of average background pixel values
        paths (list): List of image paths
        std_factor (int, optional): Standard deviation factor used to identify anomalies. Defaults value is 4.
    """    
    avg_bgs = np.array(avg_bgs)
    # calculate the standard deviation of the average background pixel values
    std = np.std(avg_bgs)
    # calculate the mean of the average background pixel values
    mean = np.mean(avg_bgs)
    for i, avg_bg in enumerate(avg_bgs):
        if abs(avg_bg - mean) > std_factor*std:
            if platform.system() == "Windows":
                print(paths[i].split("\\")[-2])
            elif platform.system() == "Linux":    
                print(paths[i].split("/")[-2])
            else:
                raise NotImplementedError("This platform is not supported yet.")    
            print("Mean: ", avg_bg)
            img = cv2.imread(paths[i])
            cv2.imshow("anomaly", img)
            cv2.waitKey(0)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # apply Gaussian blur then Otsu's binary thresholding
            #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            #thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # display the thresholded image
            #cv2.imshow("thresh", thresh)
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Do you want to remove this image? (Y,n)")
            if input() in ['', 'Y', 'y']:
                if not os.path.exists("anomalies"):
                    os.mkdir("anomalies")
                if platform.system() == "Windows":
                    dir = os.path.join("anomalies", paths[i].split("\\")[-2])
                elif platform.system() == "Linux":
                    dir = os.path.join("anomalies", paths[i].split("/")[-2])
                else:
                    raise NotImplementedError("This platform is not supported yet.")
                if not os.path.exists(dir):
                    os.mkdir(dir)
                if platform.system() == "Windows":
                    shutil.move(paths[i], os.path.join(dir, paths[i].split("\\")[-1]))
                elif platform.system() == "Linux":
                    shutil.move(paths[i], dir + "/" + paths[i].split("/")[-1])
                else:
                    raise NotImplementedError("This platform is not supported yet.")
                print("Image successfully moved.")
            print()


def copy_anomalies(avg_bgs, paths, std_factor=4):
    """This function provides a way to copy the images that are identified as anomalies to a new directory.

    Args:
        avg_bgs (list): List of average background pixel values
        paths (list): List of image paths
        std_factor (int, optional): Standard deviation factor used to identify anomalies. Defaults value is 4.
    
    Returns:
        dictionary: Dictionary of image names and their paths
    """    
    avg_bgs = np.array(avg_bgs)
    # calculate the standard deviation of the average background pixel values
    std = np.std(avg_bgs)
    # calculate the mean of the average background pixel values
    mean = np.mean(avg_bgs)
    anomalies = {}
    for i, avg_bg in enumerate(avg_bgs):
        if abs(avg_bg - mean) > std_factor*std:
            if not os.path.exists("anomalies"):
                os.mkdir("anomalies")
            if platform.system() == "Windows":
                dir = "anomalies\\" + paths[i].split("\\")[-2]
            elif platform.system() == "Linux":
                dir = "anomalies/" + paths[i].split("/")[-2]
            else:
                raise NotImplementedError("This platform is not supported yet.")    
            if not os.path.exists(dir):
                os.mkdir(dir)
            if platform.system() == "Windows":
                shutil.copy(paths[i], os.path.join(dir, paths[i].split("\\")[-1]))
                anomalies[paths[i].split("\\")[-1]] = paths[i]
            elif platform.system() == "Linux":
                shutil.copy(paths[i], os.path.join(dir, paths[i].split("/")[-1]))
                anomalies[paths[i].split("/")[-1]] = paths[i]
            else:
                raise NotImplementedError("This platform is not supported yet.")    
            print("Image successfully copied.")
    return anomalies


def delete_anomalies(anomalies):
    """This function provides a way to delete the images that are identified as anomalies.

    Args:
        anomalies (dictionary): Dictionary of image names and their paths
    """    
    for root, dirs, files in os.walk("anomalies"):
        for file in files:
            os.remove(anomalies[file])


def store_anomalies(anomalies, path="anomaly_path.json"):
    """This function provides a way to store the anomalies in a json file.

    Args:
        anomalies (dictionary): Dictionary of image names and their paths
        path (str, optional): Path to the json file. Defaults to "anomalies.json".
    """    
    with open(path, "w") as f:
        json.dump(anomalies, f)


def open_anomalies(path="anomaly_path.json"):
    """This function provides a way to open the anomalies from a json file.

    Args:
        path (str, optional): Path to the json file. Defaults to "anomalies.json".

    Returns:
        dictionary: Dictionary of image names and their paths
    """    
    with open(path, "r") as f:
        anomalies = json.load(f)
    return anomalies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script provides a way to identify anomalies in the baseline training set.")
    parser.add_argument("-s", "--source", type=str, help="Path to the baseline training set.")
    parser.add_argument("-d", "--dist", type=str, help="Path to the json file containing the distribution and associated paths.")
    parser.add_argument("-c", "--copy", type=bool, default=False, help="Copy the anomalies to a new directory.")
    parser.add_argument("-m", "--manual", type=bool, default=False, help="Manually validate the anomalies.")
    parser.add_argument("-r", "--remove", type=bool, default=False, help="Remove the anomalies.")
    parser.add_argument("--std", type=int, default=4, help="Standard deviation factor used to identify anomalies.")
    args = parser.parse_args()

    if args.source:
        avg_bgs, paths = get_background_dist(args.source)
        # store the distribution and associated paths in a json file
        store_dist("background_dist.json", avg_bgs, paths) 
    else:
        avg_bgs, paths = load_dist("background_dist.json") # requires a json file with the distribution and associated pathsn
    if args.copy:    
        anomalies = copy_anomalies(avg_bgs, paths, args.std)
        store_anomalies(anomalies)
    if args.manual:
        manually_validate_anomalies(avg_bgs, paths, args.std)
    if args.remove:
        anomalies = open_anomalies()
        delete_anomalies(anomalies)


