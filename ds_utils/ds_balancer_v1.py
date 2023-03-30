import os
import numpy as np
import shutil
import cv2
import albumentations as A
import argparse


def get_generator(): 
	"""
	Returns an albumentations.Compose object containing a list of image augmentation techniques.
	
	Returns:
		A.Compose: A composition of image augmentation techniques.
	"""
	return A.Compose([
		A.Rotate(limit=(0, 360), p=1),
		A.Flip(p=0.3),
		A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.3),
		A.IAAPerspective(scale=(0.05, 0.1), p=0.1),
		A.IAAAffine(shear=20, p=0.3),
		A.OpticalDistortion(p=0.1),
		A.RandomBrightnessContrast(p=0.2),
		A.GaussNoise(p=0.1),
		A.GaussianBlur(p=0.1),
		A.ImageCompression(quality_lower=60, p=0.1),
		A.RandomGamma(p=0.1),
		A.IAAEmboss(p=0.1),
		A.IAASharpen(p=0.1)
	])


def create_augmented_images(class_folder, num_images, generator):
	"""
	Creates augmented images for a given class folder to reach a specified number of images.
	
	Args:
		class_folder (str): The path to the folder containing images of a specific class.
		num_images (int): The desired number of images in the class folder after augmentation.
		generator (A.Compose): An albumentations.Compose object containing image augmentation techniques.
	"""
	img_list = os.listdir(class_folder)
	num_images_to_create = num_images - len(img_list)
	

	for i in range(num_images_to_create):
		img_path = os.path.join(class_folder, np.random.choice(img_list, replace=True))
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		augmented_img = generator(image=img)['image']
		output_path = os.path.join(class_folder, f'augmented_{i}.jpg')
		cv2.imwrite(output_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))


def copy_images(src, dst):	
	""" 
	Copies images from the source directory to the destination directory.

	Args:
		src (str): The path to the source directory.
		dst (str): The path to the destination directory.
	"""

	# Iterate all the files in the source directory.
	# The function should account for infinite subdirectories.
	for root, _, files in os.walk(src):
		# Iterate all the files in the current directory.
		for file in files:
			# Construct the full path of the source file.
			src_file = os.path.join(root, file)
			# Construct the full path of the destination file.
			dst_file = os.path.join(dst, root[len(src):], file)
			# Create the destination directory if it does not exist.
			if not os.path.exists(os.path.dirname(dst_file)):
				os.makedirs(os.path.dirname(dst_file))
			# Copy the file from the source to the destination.
			shutil.copyfile(src_file, dst_file)



def balance_dataset(base_dataset_path, output_path):
	"""
    Balances the dataset by creating augmented images or removing images to reach an equal number of images per class.
    
    Args:
        base_dataset_path (str): The path to the base dataset directory.
        output_path (str): The path to the directory where the balanced dataset will be stored.
    """
	# Create output directories.
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	# Copy the original images to the output directory.
	copy_images(base_dataset_path, output_path)	
	
	# Create augmentation-pipeline/generator.
	generator = get_generator()

	set_path = os.path.join(base_dataset_path, "train")
	output_set_path = os.path.join(output_path, "train")
	# The number of images per class is the total number of images in the set divided by the number of classes.
	target_num_images_per_class = sum([len(os.listdir(os.path.join(set_path, class_name))) for class_name in os.listdir(set_path)]) // len(os.listdir(set_path))

	if not os.path.exists(output_path):
		os.makedirs(output_path)
	
	# Loop through classes and balance.
	for class_name in os.listdir(set_path):
		output_class_path = os.path.join(output_set_path, class_name)

		if not os.path.exists(output_class_path):
			os.makedirs(output_class_path)

		# Number of images in the current class.
		n = len(os.listdir(output_class_path))

		# Remove images if there are too many.
		if n > target_num_images_per_class:
			for img_name in np.random.choice(os.listdir(output_class_path), size=n-target_num_images_per_class, replace=False):
				os.remove(os.path.join(output_class_path, img_name))
		# Create augmented images if there are too few.		
		elif n < target_num_images_per_class:
			create_augmented_images(output_class_path, target_num_images_per_class, generator)



def parse_arguments():
    """
    Parses command line arguments and returns an object containing the arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Balance a dataset by creating augmented images or removing images to reach an equal number of images per class. Note that this script was developed on a linux system and may not work on other operating systems.")
    parser.add_argument('--source', '-s', type=str, required=True, help='The path to the base dataset directory.')
    parser.add_argument('--destination', '-d', type=str, required=True, help='The path to the directory where the balanced dataset will be stored.')

    return parser.parse_args()

if __name__ == "__main__":
	args = parse_arguments()
	balance_dataset(args.source, args.destination)
	print("Successfully balanced dataset!")