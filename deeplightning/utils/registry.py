#from deeplightning.utils.metrics import metric_accuracy
from deeplightning.utils.metrics import Metric_Accuracy, Metric_ConfusionMatrix, Metric_PrecisionRecallCurve


__TaskRegistry__ = [
    # Image
    "ImageClassification",
    "ImageReconstruction",
    "ObjectDetection",
    # Audio
    "AudioClassification",
]

__MetricsRegistry__ = {
    # Image
	"ImageClassification": {
        "Accuracy_train": Metric_Accuracy,
        "Accuracy_val": Metric_Accuracy,
        "Accuracy_test": Metric_Accuracy,
        "ConfusionMatrix_val": Metric_ConfusionMatrix,
        "ConfusionMatrix_test": Metric_ConfusionMatrix,
		"PrecisionRecallCurve_val": Metric_PrecisionRecallCurve,
		"PrecisionRecallCurve_test": Metric_PrecisionRecallCurve,
	},
    "ImageReconstruction": {
        "_": None,
    },
    "ObjectDetection": {
        "_": None,
    },
    # Audio
    "AudioClassification":{
        "Accuracy": Metric_Accuracy,
        "ConfusionMatrix": Metric_ConfusionMatrix,
		"PrecisionRecallCurve": Metric_PrecisionRecallCurve,
    },
}

__LoggerRegistry__ = [
    "wandb",
]
