from PIL import Image
import os
import shutil


def check_image(root_dir, dest_dir):
    """Check a directory, and subfolders, for broken images and move them to a new directory

    Args:
        root_dir (string): string path to the directory to check
        dest_dir (string): string path to the directory to move broken images to
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    im = Image.open(os.path.join(root, file))
                    im.verify()
                except:
                    shutil.move(os.path.join(root, file), dest_dir)


if __name__ == '__main__':
    root_dir = "path/to/root/dir"
    dest_dir = "path/to/dest/dir"
    check_image(root_dir, dest_dir)
    print("Done")