import os
import numpy as np
import shutil
import albumentations as A
import cv2

def get_generator():
    # Set up data augmentation for images using Albumentations
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])


# Function to create augmented images
def create_augmented_images(class_folder, num_images, generator):
    img_list = os.listdir(class_folder)
    num_images_to_create = num_images - len(img_list)
    
    if num_images_to_create <= 0:
        return

    for i in range(num_images_to_create):
        img_path = os.path.join(class_folder, np.random.choice(img_list))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented_img = generator(image=img)['image']
        output_path = os.path.join(class_folder, f'augmented_{i}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))


def balance_dataset(base_dataset_path, output_path):
    # Create output directories
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generator = get_generator()
    for set in os.listdir(base_dataset_path):
        set_path = os.path.join(base_dataset_path, set)
        output_set_path = os.path.join(output_path, set)
        target_num_images_per_class = sum([len(os.listdir(os.path.join(set_path, class_name))) for class_name in os.listdir(set_path)]) // len(os.listdir(set_path))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Loop through classes and balance
        for class_name in os.listdir(set_path):
            class_path = os.path.join(set_path, class_name)
            output_class_path = os.path.join(output_set_path, class_name)

            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            # Number of images
            n = 0
            # Copy original images to the output folder
            for img_name in os.listdir(class_path):
                src = os.path.join(class_path, img_name)
                dst = os.path.join(output_class_path, img_name)
                shutil.copyfile(src, dst)
                n += 1 

            img_list = os.listdir(output_class_path)

            if n > target_num_images_per_class:
                # Remove extra images
                for img_name in np.random.choice(img_list, size=n-target_num_images_per_class, replace=False):
                    os.remove(os.path.join(output_class_path, img_name))
            elif n < target_num_images_per_class:
                # Create augmented images
                create_augmented_images(output_class_path, target_num_images_per_class, generator)




if __name__ == "__main__":
    # Change these to the appropriate paths
    base_dataset_path = 'dataset'
    output_path = 'balanced_dataset'
    balance_dataset(base_dataset_path, output_path)
