from prediction import Prediction

predict = Prediction(
    predictor_file="shape_predictor.dat",
    detector_file="object_detector.svm",
    upsample_num=0,
    threshold=30,
)

print(predict)
