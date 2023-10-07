from dlib import (
    simple_object_detector_training_options,
    shape_predictor_training_options,
)
from omegaconf import DictConfig, OmegaConf


def object_detection_training_options(cfg: DictConfig):
    # * ======== Object Detection Trainer
    options = simple_object_detector_training_options()
    args = cfg["object_detection"]
    options.add_left_right_image_flips = args["symmetrical"]
    options.C = args["C"]
    options.num_threads = args["num_threads"]
    options.be_verbose = True
    options.epsilon = args["epsilon"]
    options.upsample_limit = args["upsample_limit"]

    if hasattr(args, "window_size"):
        options.detection_window_size = args["window_size"]

    return options


def shape_prediction_training_options(cfg: DictConfig):
    options = shape_predictor_training_options()
    args = cfg["shape_predictor"]
    options.num_trees_per_cascade_level = args["num_trees_per_cascade"]
    options.nu = args["regularization"]
    options.num_threads = args["num_threads"]
    options.tree_depth = args["tree_depth"]
    options.cascade_depth = args["cascade_depth"]
    options.feature_pool_size = args["feature_pool_size"]
    options.num_test_splits = args["test_splits"]
    options.oversampling_amount = args["oversampling"]
    options.be_verbose = True

    return options
