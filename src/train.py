import hydra
import wandb
from omegaconf import DictConfig

from utils.config_to_train_options import (
    object_detection_training_options,
    shape_prediction_training_options,
)
import dlib


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig):
    wandb.init(project="Quadricarinatus", config=dict(cfg))
    # TODO: pretty print cfg

    object_options = object_detection_training_options(cfg)
    dlib.train_simple_object_detector(
        cfg["train"], f"out/{cfg['output_name']}object_detection.svm", object_options
    )

    shape_options = shape_prediction_training_options(cfg)
    dlib.train_shape_predictor(
        cfg["train"], f"out/{cfg['output_name']}shape_predictor.dat", shape_options
    )

    # Test

    object_training_performance = dlib.test_simple_object_detector(
        cfg["train"], f"out/{cfg['output_name']}object_detection.svm"
    )
    object_validation_performance = dlib.test_simple_object_detector(
        cfg["valid"], f"out/{cfg['output_name']}object_detection.svm"
    )

    shape_training_performance = dlib.test_shape_predictor(
        cfg["train"], f"out/{cfg['output_name']}shape_predictor.dat"
    )

    shape_validation_performance = dlib.test_shape_predictor(
        cfg["valid"], f"out/{cfg['output_name']}shape_predictor.dat"
    )

    # Log

    logs = {
        "object_training": {
            "average_precision": object_training_performance.average_precision,
            "precision": object_training_performance.precision,
            "recall": object_training_performance.recall,
        },
        "object_validation": {
            "average_precision": object_validation_performance.average_precision,
            "precision": object_validation_performance.precision,
            "recall": object_validation_performance.recall,
        },
        "shape_training_error": shape_training_performance,
        "shape_validation_error": shape_validation_performance,
    }

    wandb.log(logs)
    print(logs)


run()
