import os
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root, "heart_disease_model.pkl")
onnx_output = os.path.join(os.path.dirname(__file__), "heart_disease_model.onnx")

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    initial_type = [("float_input", FloatTensorType([None, 13]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(onnx_output, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"[SUCCESS] Exported ONNX model to {onnx_output}")
else:
    print(f"[WARNING] Model not found at {model_path}")
