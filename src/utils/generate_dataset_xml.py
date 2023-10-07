import numpy as np

from xml.etree.ElementTree import Element, tostring, ElementTree
from xml.dom import minidom
import os


def read_csv(file_name):
    """
    This function reads a XY coordinate file (following the tpsDig coordinate system) containing several specimens(rows)
    and any number of landmarks. It is generally assumed here that the file contains a header and no other
    columns other than an id column (first column) and the X0 Y0 ...Xn Yn coordinates for n landmarks.It is also
    assumed that the file contains no missing values.

    Parameters:
        file_name (str): The XY coordinate file (csv format)
    Returns:
        dict: dictionary containing two keys (im= image id, coords= array with 2D coordinates)

    """
    csv_file = open(file_name, "r")
    csv = csv_file.read().splitlines()
    csv_file.close()
    im, coords_array = [], []

    for i, ln in enumerate(csv):
        if i > 0:
            im.append(ln.split(",")[0])
            coord_vec = ln.split(",")[1:]
            coords_mat = np.reshape(coord_vec, (int(len(coord_vec) / 2), 2))
            coords = np.array(coords_mat, dtype=float)
            coords_array.append(coords)
    return {"im": im, "coords": coords_array}


def read_tps(file_name):
    """
    This function reads a tps coordinate file containing several specimens and an arbitrary number of landmarks.
    A single image file can contain as many specimens as wanted.
    It is generally assumed here that all specimens were landmarked in the same order.It is also  assumed that
    the file contains no missing values.

    Parameters:
        file_name (str): The tps coordinate file
    Returns:
        dict: dictionary containing four keys
        (lm= number of landmarks,im= image id, scl= scale, coords= array with 2D coordinates)

    """
    tps_file = open(file_name, "r")
    tps = tps_file.read().splitlines()
    tps_file.close()
    lm, im, sc, coords_array = [], [], [], []

    for i, ln in enumerate(tps):
        if ln.startswith("LM"):
            lm_num = int(ln.split("=")[1])
            lm.append(lm_num)
            coords_mat = []
            for j in range(i + 1, i + 1 + lm_num):
                coords_mat.append(tps[j].split(" "))
            coords_mat = np.array(coords_mat, dtype=float)
            coords_array.append(coords_mat)

        if ln.startswith("IMAGE"):
            im.append(ln.split("=")[1])

        if ln.startswith("SCALE"):
            sc.append(ln.split("=")[1])
    return {"lm": lm, "im": im, "scl": sc, "coords": coords_array}


def generate_dlib_xml(images, sizes, folder="train", out_file="output.xml"):
    """
    Generates a dlib format xml file for training or testing of machine learning models.

    Parameters:
        out_file: output file name
        images (dict): dictionary output by read_tps or read_csv functions
        sizes (dict)= dictionary of image file sizes output by the split_train_test function
        folder(str)= name of the folder containing the images
    """
    root = Element("dataset")
    # root.append(Element("name"))
    # root.append(Element("comment"))

    images_e = Element("images")
    root.append(images_e)

    for i in range(0, len(images["im"])):
        name = os.path.splitext(images["im"][i])[0] + ".jpg"
        path = os.path.join(folder, name)
        if os.path.isfile(path) is True:
            present_tags = []
            for img in images_e.findall("image"):
                present_tags.append(img.get("file"))

            if path in present_tags:
                pos = present_tags.index(path)
                images_e[pos].append(
                    add_bbox_element(images["coords"][i], sizes[name][0])
                )

            else:
                images_e.append(
                    add_image_element(name, images["coords"][i], sizes[name][0], path)
                )

    et = ElementTree(root)
    xml_str = minidom.parseString(tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out_file, "w") as f:
        f.write(xml_str)


def add_part_element(bbox, num, sz):
    """
    Internal function used by generate_dlib_xml. It creates a 'part' xml element containing the XY coordinates
    of an arbitrary number of landmarks. Parts are nested within boxes.

    Parameters:
        bbox (array): XY coordinates for a specific landmark
        num(int)=landmark id
        sz (int)=the image file's height in pixels


    Returns:
        part (xml tag): xml element containing the 2D coordinates for a specific landmark id(num)

    """
    part = Element("part")
    part.set("name", str(int(num)))
    part.set("x", str(int(bbox[0])))
    part.set("y", str(int(sz - bbox[1])))
    return part


def add_bbox_element(bbox, sz, padding=0):
    """
    Internal function used by generate_dlib_xml. It creates a 'bounding box' xml element containing the
    four parameters that define the bounding box (top,left, width, height) based on the minimum and maximum XY
    coordinates of its parts(i.e.,landmarks). An optional padding can be added to the bounding box.Boxes are
    nested within images.

    Parameters:
        bbox (array): XY coordinates for all landmarks within a bounding box
        sz (int)= the image file's height in pixels
        padding(int)= optional parameter definining the amount of padding around the landmarks that should be
                       used to define a bounding box, in pixels (int).


    Returns:
        box (xml tag): xml element containing the parameters that define a bounding box and its corresponding parts

    """
    box = Element("box")
    height = bbox[:, 1].max() - bbox[:, 1].min() + 2 * padding
    width = bbox[:, 0].max() - bbox[:, 0].min() + 2 * padding
    top = sz - bbox[:, 1].max() - padding
    if top < 1:
        top = 1
    left = bbox[:, 0].min() - padding
    if left < 1:
        left = 1

    box.set("top", str(int(top)))
    box.set("left", str(int(left)))
    box.set("width", str(int(width)))
    box.set("height", str(int(height)))
    for i in range(0, len(bbox)):
        box.append(add_part_element(bbox[i, :], i, sz))
    return box


def add_image_element(image, coords, sz, path):
    """
    Internal function used by generate_dlib_xml. It creates a 'image' xml element containing the
    image filename and its corresponding bounding boxes and parts.

    Parameters:
        image (str): image filename
        coords (array)=  XY coordinates for all landmarks within a bounding box
        sz (int)= the image file's height in pixels


    Returns:
        image (xml tag): xml element containing the parameters that define each image (boxes+parts)

    """
    image_e = Element("image")
    image_e.set("file", str(path))
    image_e.append(add_bbox_element(coords, sz))
    return image_e
