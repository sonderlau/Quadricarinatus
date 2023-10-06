import optuna

import dlib


def run_optimization():
    # Object optimization
    # object_optimization()

    shape_optimization()


def shape_optimization(num: int = 10):
    study = optuna.create_study(direction="minimize")
    study.optimize(shape_objective, n_trials=num, timeout=300, show_progress_bar=True)

    print(study.best_trials)


def shape_objective(trial: optuna.trial.Trial):
    options = dlib.shape_predictor_training_options()
    options.num_trees_per_cascade_level = 10
    options.nu = 0.1
    options.num_threads = 2
    options.tree_depth = trial.suggest_int("tree_depth", 1, 8)
    options.cascade_depth = 15
    options.feature_pool_size = trial.suggest_int("feature_pool_size", 30, 10000)
    options.oversampling_amount = trial.suggest_int("oversampling_amount", 1, 100)
    options.be_verbose = False

    dlib.train_shape_predictor("dataset/train.xml", "out/shape_predictor.dat", options)
    shape_training_error = dlib.test_shape_predictor(
        "dataset/train.xml", "out/shape_predictor.dat"
    )

    return float(shape_training_error)


def object_optimization(num: int = 10):
    study = optuna.create_study(directions=["maximize", "maximize", "maximize"])
    study.optimize(object_objective, n_trials=num, timeout=300)

    print(study.best_trials)


def object_objective(trial: optuna.trial.Trial):
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = (
        False if trial.suggest_int("flip", 0, 1) == 0 else True
    )
    options.C = trial.suggest_float("C", 1e-10, 1e10, log=True)
    options.num_threads = 2
    options.be_verbose = False
    options.epsilon = trial.suggest_float("epsilon", 1e-10, 1e10, log=True)
    options.upsample_limit = trial.suggest_int("upsample_limit", 0, 3)

    dlib.train_simple_object_detector(
        "dataset/train.xml", "out/object_detection.svm", options
    )
    object_training_performance = dlib.test_simple_object_detector(
        "dataset/train.xml", "out/object_detection.svm"
    )
    average_precision = object_training_performance.average_precision
    precision = object_training_performance.precision
    recall = object_training_performance.recall

    return average_precision, precision, recall


run_optimization()
