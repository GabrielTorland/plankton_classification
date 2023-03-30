import numpy as np
import cv2
from scipy import stats as st
import os
import argparse
import platform

# set current directory to the directory of the script
os.chdir(os.path.dirname(__file__))



def check_aspect_ratio(image):
    '''Check the aspect ratio of the image.
    Args:
        image (numpy.ndarray): A numpy array of shape [height, width, channels].
    Returns:
        aspect_ratio (float): The aspect ratio of the image.
    '''
    height, width = image.shape[0], image.shape[1]
    aspect_ratio = height / width
    return aspect_ratio


def add_padding(image, bg_color):
    '''
    Add padding to the image.
    Args:
        image (numpy.ndarray): A numpy array of shape [height, width, channels].
        bg_color (tuple): The background color values of the image as a BGR tuple.
    Returns:
        image (numpy.ndarray): A square image padded with the background color.
    '''
    # get image dimensions
    height, width = image.shape[0], image.shape[1]
    # get the difference between the dimensions
    diff = height - width
    

    if diff > 0:
        if diff % 2 == 1:
            pad = diff // 2
            image = cv2.copyMakeBorder(image, 0, 0, pad+1, pad, cv2.BORDER_CONSTANT, value=bg_color)
        else:
            pad = diff // 2
            image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=bg_color)
    elif diff < 0:
        if diff % 2 == 1:
            pad = -diff // 2
            image = cv2.copyMakeBorder(image, pad+1, pad, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)
        else:    
            pad = -diff // 2
            image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)
    return image

    
def edge_pixel_values(image):
    '''Get the edge pixel values of the image and find the mode value for each channel.
    Args:
        image: numpy.ndarray of shape [height, width, channels].
    Returns:
        A tuple of the mode edge pixel int values for each channel (BGR).
    '''
    b, g, r = cv2.split(image)
    b_top = b[0]
    b_bottom = b[-1]
    b_left = b[:, 0]
    b_right = b[:, -1]

    g_top = g[0]
    g_bottom = g[-1]
    g_left = g[:, 0]
    g_right = g[:, -1]

    r_top = r[0]
    r_bottom = r[-1]
    r_left = r[:, 0]
    r_right = r[:, -1]

    b_tot = np.concatenate((b_top, b_bottom, b_left, b_right))
    g_tot = np.concatenate((g_top, g_bottom, g_left, g_right))
    r_tot = np.concatenate((r_top, r_bottom, r_left, r_right))

    b_mode = st.mode(b_tot)
    g_mode = st.mode(g_tot)
    r_mode = st.mode(r_tot)

    bgr_mode_values = (int(b_mode[0][0]), int(g_mode[0][0]), int(r_mode[0][0]))
    
    return bgr_mode_values

def find_preprocess_images(path):
    '''Find all images wiht aspect ratio other than one, in the given directory.
    Args:
        path (str): A string representing the path to the directory.
    Returns:    
        img_paths (dict): A dictionary of the image names and their paths.
    '''
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.jpg'): continue
            img = cv2.imread(os.path.join(root, file))
            asp_ratio = check_aspect_ratio(img)
            # if the aspect ratio is not 1.0, then add padding and copy to new directory
            if asp_ratio == 1: continue
            bgr_mode_values = edge_pixel_values(img)
            img_padded = add_padding(img, bgr_mode_values)
            # determine if the os is windows or linux    
            cv2.imwrite(os.path.join(root, file), img_padded)
                
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pads the images with aspect ration other than 1.0")
    parser.add_argument("-s", "--source", type=str, help="Path to the directory containing the images")
    args = parser.parse_args()
    find_preprocess_images("dataset")




    
