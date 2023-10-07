import os
import random
import shutil

import cv2


def split_train_test(input_dir):
    """
    Splits an image directory into 'train' and 'test' directories. The original image directory is preserved.
    When creating the new directories, this function converts all image files to 'jpg'. The function returns
    a dictionary containing the image dimensions in the 'train' and 'test' directories.

    Parameters:
        input_dir(str)=original image directory

    Returns:
        sizes (dict): dictionary containing the image dimensions in the 'train' and 'test' directories.
    """
    # Listing the filenames.Folders must contain only image files (extension can vary).Hidden files are ignored
    filenames = os.listdir(input_dir)
    filenames = [os.path.join(input_dir, f) for f in filenames if not f.startswith(".")]

    # Splitting the images into 'train' and 'test' directories (80/20 split)
    random.seed(845)
    filenames.sort()
    random.shuffle(filenames)
    split = int(0.8 * len(filenames))
    train_set = filenames[:split]
    test_set = filenames[split:]

    filenames = {"train": train_set, "test": test_set}
    sizes = {}
    for split in ["train", "test"]:
        sizes[split] = {}
        if not os.path.exists(split):
            os.mkdir(split)
        else:
            print(
                "Warning: the folder {} already exists. It's being replaced".format(
                    split
                )
            )
            shutil.rmtree(split)
            os.mkdir(split)

        for filename in filenames[split]:
            basename = os.path.basename(filename)
            name = os.path.splitext(basename)[0] + ".jpg"
            sizes[split][name] = image_prep(filename, name, split)
    return sizes


def image_prep(file, name, dir_path):
    """
    Internal function used by the split_train_test function. Reads the original image files and, while
    converting them to jpg, gathers information on the original image dimensions.

    Parameters:
        file(str)=original path to the image file
        name(str)=basename of the original image file
        dir_path(str)= directory where the image file should be saved to

    Returns:
        file_sz(array): original image dimensions
    """
    img = cv2.imread(file)
    if img is None:
        print("File {} was ignored".format(file))
    else:
        file_sz = [img.shape[0], img.shape[1]]
        cv2.imwrite(os.path.join(dir_path, name), img)
        return file_sz
