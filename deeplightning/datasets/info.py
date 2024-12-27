from dataclasses import dataclass, field



@dataclass
class MNIST_INFO:
    dataset_name: str = "MNIST"
    image_size: tuple[int] = (28, 28)  # (width,height)
    num_channels: int = 1
    num_classes: int = 10
    normalization_constants: dict = field(
        default_factory = lambda: {
            'mean': [0.1307],
            'std': [0.3081]
        }
    )


@dataclass
class HAM10000_INFO:
    dataset_name: str = "HAM10000"
    image_size: tuple[int] = (600, 450)  # (width, height)
    num_channels: int = 3
    num_classes: dict = field(
        default_factory = lambda: {
            "image_classification": 7,
            "image_semantic_segmentation": 2,
        }
    )
    normalization_constants: dict = field(
        default_factory = lambda: {
            "mean": [0.7639, 0.5463, 0.5703],
            "std": [0.0870, 0.1155, 0.1295],
        }
    )


DATASET_INFO = {
    "MNIST": MNIST_INFO,
    "HAM10000": HAM10000_INFO,
}