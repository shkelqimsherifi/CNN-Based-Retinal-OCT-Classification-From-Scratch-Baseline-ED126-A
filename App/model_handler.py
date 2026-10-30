import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras

from custom_layers import CUSTOM_OBJECTS

IMG_SIZE = (224, 224)
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = (
    APP_DIR / "eye_trained_model.keras",
    APP_DIR / "eye_trained_model.h5",
)


@dataclass
class LoadedModel:
    name: str
    path: Path
    model: object


class ModelHandler:
    def __init__(self):
        self.models = []
        self.model = None
        self.model_path = None
        self.class_names = CLASS_NAMES

    def load_model(self, model_path, append=False):
        """Loads a Keras model from the given path."""
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            return False, f"Model file not found: {model_path}"

        try:
            model = keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects=CUSTOM_OBJECTS,
            )
            loaded_model = LoadedModel(model_path.stem, model_path, model)

            if append:
                self.models.append(loaded_model)
            else:
                self.models = [loaded_model]

            self._sync_active_model()
            return True, "Model loaded successfully."
        except Exception as e:
            if not append:
                self.models = []
                self._sync_active_model()
            return False, f"{type(e).__name__}: {e}"

    def load_models(self, model_paths):
        """Loads several models and replaces the current model list."""
        loaded_models = []
        errors = []

        for model_path in model_paths:
            model_path = Path(model_path).expanduser().resolve()
            if not model_path.exists():
                errors.append(f"{model_path.name}: file not found")
                continue

            try:
                model = keras.models.load_model(
                    str(model_path),
                    compile=False,
                    custom_objects=CUSTOM_OBJECTS,
                )
                loaded_models.append(LoadedModel(model_path.stem, model_path, model))
            except Exception as e:
                errors.append(f"{model_path.name}: {type(e).__name__}: {e}")

        if not loaded_models:
            self.models = []
            self._sync_active_model()
            return False, "No models loaded. " + " | ".join(errors)

        self.models = loaded_models
        self._sync_active_model()

        if errors:
            return True, "Some models loaded. " + " | ".join(errors)
        return True, f"Loaded {len(loaded_models)} model(s)."

    def load_default_model(self):
        """Loads the bundled model saved next to the application files."""
        for model_path in DEFAULT_MODEL_CANDIDATES:
            if model_path.exists():
                return self.load_model(model_path)

        candidates = ", ".join(str(path) for path in DEFAULT_MODEL_CANDIDATES)
        return False, f"No bundled model found. Checked: {candidates}"

    def predict(self, image_path):
        """Predicts with the first loaded model for backward compatibility."""
        if not self.models:
            return None, "Model not loaded."
        return self.predict_with_model(self.models[0], image_path)

    def predict_with_model(self, loaded_model, image_path):
        """Preprocesses the image and predicts with a specific loaded model."""
        try:
            input_arr = self._preprocess_image(image_path)
            prediction = loaded_model.model.predict(input_arr, verbose=0)
            result_index = int(np.argmax(prediction))

            label = self.class_names[result_index]
            confidence = float(np.max(prediction))
            probabilities = {
                class_name: float(prediction[0][index])
                for index, class_name in enumerate(self.class_names)
            }

            return {
                "label": label,
                "confidence": confidence,
                "index": result_index,
                "probabilities": probabilities,
            }, None
        except Exception as e:
            return None, str(e)

    def _preprocess_image(self, image_path):
        """Load an OCT image as grayscale and match the notebook input pipeline."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32)

        # The saved model already contains a Rescaling(1/255) layer.
        return np.expand_dims(img_float[..., np.newaxis], axis=0)

    def _sync_active_model(self):
        if self.models:
            self.model = self.models[0].model
            self.model_path = self.models[0].path
        else:
            self.model = None
            self.model_path = None
