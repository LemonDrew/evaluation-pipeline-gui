import time
import numpy as np
from PIL import Image
 
 
# ── Supported torchvision architectures ──────────────────────────────────────
ARCHITECTURES = {
    "ResNet-18":       ("torchvision.models", "resnet18"),
    "ResNet-34":       ("torchvision.models", "resnet34"),
    "ResNet-50":       ("torchvision.models", "resnet50"),
    "ResNet-101":      ("torchvision.models", "resnet101"),
    "MobileNetV2":     ("torchvision.models", "mobilenet_v2"),
    "MobileNetV3-S":   ("torchvision.models", "mobilenet_v3_small"),
    "MobileNetV3-L":   ("torchvision.models", "mobilenet_v3_large"),
    "EfficientNet-B0": ("torchvision.models", "efficientnet_b0"),
    "EfficientNet-B1": ("torchvision.models", "efficientnet_b1"),
    "EfficientNet-B2": ("torchvision.models", "efficientnet_b2"),
}
 
 
class ModelRunner:
    """
    Unified inference interface for .tflite and .pt (torchvision state-dict) models.
 
    Usage:
        runner = ModelRunner(path, arch_key)   # arch_key only needed for .pt
        runner.load()
        h, w   = runner.input_size()           # (height, width)
        result = runner.predict(pil_image)     # {"label": int, "time_ms": float}
        runner.close()
    """
 
    def __init__(self, model_path: str, arch_key: str | None = None):
        self.model_path = model_path
        self.arch_key   = arch_key
        self._kind      = None   # "tflite" | "pytorch"
        self._model     = None
        self._input_details  = None
        self._output_details = None
 
    # ── Loading ───────────────────────────────────────────────────────────────
 
    def load(self):
        if self.model_path.endswith(".tflite"):
            self._load_tflite()
        elif self.model_path.endswith(".pt"):
            self._load_pytorch()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
 
    def _load_tflite(self):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=self.model_path)
        interp.allocate_tensors()
        self._model          = interp
        self._input_details  = interp.get_input_details()
        self._output_details = interp.get_output_details()
        self._kind           = "tflite"
 
    def _load_pytorch(self):
        import importlib
        import torch
 
        if self.arch_key is None or self.arch_key not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{self.arch_key}'. "
                f"Valid keys: {list(ARCHITECTURES)}"
            )
 
        module_name, fn_name = ARCHITECTURES[self.arch_key]
        module = importlib.import_module(module_name)
        model_fn = getattr(module, fn_name)
 
        state_dict = torch.load(self.model_path, map_location="cpu")
        # Unwrap common wrappers
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
 
        model = model_fn(weights=None)
        model.load_state_dict(state_dict)
        model.eval()
 
        self._model = model
        self._kind  = "pytorch"
 
    # ── Input shape ───────────────────────────────────────────────────────────
 
    def input_size(self) -> tuple[int, int]:
        """Return (height, width) of the model's expected input."""
        if self._kind == "tflite":
            shape = self._input_details[0]["shape"]   # [1, H, W, C]
            return int(shape[1]), int(shape[2])
        elif self._kind == "pytorch":
            # Standard torchvision models all accept 224×224 by default.
            # If the state dict encodes a different size it can't be inferred
            # without a forward pass, so we use 224 as the safe default.
            return 224, 224
 
    # ── Inference ─────────────────────────────────────────────────────────────
 
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a pre-resized PIL image.
        Returns {"label": int, "time_ms": float}
        """
        if self._kind == "tflite":
            return self._predict_tflite(image)
        elif self._kind == "pytorch":
            return self._predict_pytorch(image)
 
    def _predict_tflite(self, image: Image.Image) -> dict:
        import tensorflow as tf
 
        dtype = self._input_details[0]["dtype"]
        arr   = np.array(image, dtype=np.float32)
 
        # Normalise: quantised models expect uint8 [0,255], float models [0,1]
        if dtype == np.uint8:
            arr = arr.astype(np.uint8)
        else:
            arr = arr / 255.0
 
        arr = np.expand_dims(arr, axis=0).astype(dtype)
        self._model.set_tensor(self._input_details[0]["index"], arr)
 
        t0 = time.perf_counter()
        self._model.invoke()
        t1 = time.perf_counter()
 
        output = self._model.get_tensor(self._output_details[0]["index"])
        label  = int(np.argmax(output[0]))
        return {"label": label, "time_ms": (t1 - t0) * 1000}
 
    def _predict_pytorch(self, image: Image.Image) -> dict:
        import torch
 
        arr    = np.array(image, dtype=np.float32) / 255.0          # [H,W,3]
        mean   = np.array([0.485, 0.456, 0.406])
        std    = np.array([0.229, 0.224, 0.225])
        arr    = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
 
        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model(tensor)
        t1 = time.perf_counter()
 
        label = int(torch.argmax(output, dim=1).item())
        return {"label": label, "time_ms": (t1 - t0) * 1000}
 
    # ── Cleanup ───────────────────────────────────────────────────────────────
 
    def close(self):
        self._model = None