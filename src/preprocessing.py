import argparse
import os
from utils.preprocess_dataset import split_train_test
from utils.generate_dataset_xml import read_csv, read_tps, generate_dlib_xml

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input-dir",
    type=str,
    default="images",
    help="input directory containing image files (default = images)",
    metavar="",
)
ap.add_argument(
    "-c",
    "--csv-file",
    type=str,
    default=None,
    help="(optional) XY coordinate file in csv format",
    metavar="",
)
ap.add_argument(
    "-t",
    "--tps-file",
    type=str,
    default=None,
    help="(optional) tps coordinate file",
    metavar="",
)

args = vars(ap.parse_args())

# 检查文件夹是否存在
assert os.path.isdir(args["input_dir"]), "Could not find the folder {}".format(
    args["input_dir"]
)

# 创建 dataset 文件夹
if not os.path.exists("dataset"):
    os.mkdir("dataset")

# 分割数据集
file_sizes = split_train_test(args["input_dir"])


# csv 文件
if args["csv_file"] is not None:
    dict_csv = read_csv(args["csv_file"])
    generate_dlib_xml(
        dict_csv, file_sizes["train"], folder="train", out_file="dataset/train.xml"
    )
    generate_dlib_xml(
        dict_csv, file_sizes["test"], folder="test", out_file="dataset/test.xml"
    )

# tps 文件
if args["tps_file"] is not None:
    dict_tps = read_tps(args["tps_file"])
    generate_dlib_xml(
        dict_tps, file_sizes["train"], folder="train", out_file="dataset/train.xml"
    )
    generate_dlib_xml(
        dict_tps, file_sizes["test"], folder="test", out_file="dataset/test.xml"
    )
