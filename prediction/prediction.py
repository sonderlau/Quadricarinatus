import dlib


class Prediction:
    def __init__(
        self,
        predictor_file: str,
        detector_file: str,
        upsample_num: int = 0,
        threshold: int = 0,
    ):
        self.predictor = dlib.shape_predictor(predictor_file)
        self.detector = dlib.fhog_object_detector(detector_file)
        self.upsample_num = upsample_num
        self.threshold = threshold

    def predict(self, img):
        """

        Args:
            img: Opencv Image

        Returns:
            results: []
        """
        results = []

        [boxes, confidences, _] = dlib.fhog_object_detector.run(
            self.detector,
            img,
            upsample_num_times=self.upsample_num,
            adjust_threshold=self.threshold,
        )

        # TODO: 置信度
        print(confidences)

        for k, d in enumerate(boxes):
            shape = self.predictor(img, d)

            box_result = {
                "left": d.left(),
                "top": d.top(),
                "bottom": d.bottom(),
                "right": d.right(),
                "key_points": [],
            }

            for i in range(0, shape.num_parts):
                box_result["key_points"].append(
                    {"id": i, "x": shape.part(i).x, "y": shape.part(i).y}
                )

            results.append(box_result)
        return results
