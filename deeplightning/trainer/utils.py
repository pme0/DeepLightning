
def process_model_outputs(outputs, model):
    """Processes model outouts and selects the appropriate elements
    """

    if model.__class__.__name__ == "DeepLabV3":
        # `DeepLabV3` returns a dictionaty with keys `out` (segmentation mask)
        # and optionally `aux` if an auxiliary classifier is used.
        return outputs["out"]