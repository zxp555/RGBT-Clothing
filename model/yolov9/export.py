from types import SimpleNamespace


def export_formats():
    # Minimal format table for DetectMultiBackend._model_type().
    return SimpleNamespace(
        Suffix=[
            ".pt",
            ".torchscript",
            ".onnx",
            "_end2end.onnx",
            "_openvino_model",
            ".engine",
            ".mlmodel",
            "_saved_model",
            ".pb",
            ".tflite",
            "_edgetpu.tflite",
            "_web_model",
            "_paddle_model",
        ]
    )
