import os
import shutil



def copy_jpg_from_root_dir(root_dir, dest_dir):
    """
    Copies all jpg files from root directory, and subfolders, and sorts them into folders based on the name of the subfolder they were copied from.

    Args:
        root_dir (string): location of the root folder that contains the subfolders
        dest_dir (string): location of the destination folder where the jpg files will be copied to
    """    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                folder_name = os.path.basename(root)
                dest_path = os.path.join(dest_dir, folder_name)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.copy(os.path.join(root, file), dest_path)


if __name__ == '__main__':
    root_dir = "path\to\root"
    dest_dir = "path\to\destination"
    copy_jpg_from_root_dir(root_dir, dest_dir)
    