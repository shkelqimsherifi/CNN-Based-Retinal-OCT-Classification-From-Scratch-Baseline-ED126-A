import os
import tensorflow as tf
from model_handler import ModelHandler
import numpy as np

def test_model_handler():
    handler = ModelHandler()
    
    # Check if trained_model2.keras exists
    model_path = "eye_trained_model.keras"
    if not os.path.exists(model_path):
        print(f"FAILED: {model_path} not found.")
        return

    # Test loading
    success, msg = handler.load_model(model_path)
    if success:
        print("SUCCESS: Model loaded.")
    else:
        print(f"FAILED: Model load error: {msg}")
        return

    # Test preprocessing and architecture
    if handler.model:
        input_shape = handler.model.input_shape
        print(f"Model Input Shape: {input_shape}")
        if input_shape == (None, 224, 224, 1):
            print("SUCCESS: Correct input shape.")
        else:
            print(f"WARNING: Unexpected input shape {input_shape}. Application expects 224x224.")

if __name__ == "__main__":
    test_model_handler()
    print("Testing complete.")
