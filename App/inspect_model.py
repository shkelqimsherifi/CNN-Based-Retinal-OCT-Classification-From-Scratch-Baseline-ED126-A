import zipfile
import h5py
import json
import os

def inspect_keras_zip(filepath):
    print(f"--- Inspecting {filepath} as ZIP ---")
    if not zipfile.is_zipfile(filepath):
        print("Not a valid ZIP file.")
        return False
    
    with zipfile.ZipFile(filepath, 'r') as z:
        print("ZIP Contents:")
        for name in z.namelist():
            print(f"  - {name}")
        
        # Try to read metadata or config
        if 'config.json' in z.namelist():
            try:
                config_content = z.read('config.json').decode('utf-8')
                config = json.loads(config_content)
                keras_version = config.get('keras_version', 'unknown')
                print(f"Keras Version from config.json: {keras_version}")
                class_name = config.get('class_name', 'unknown')
                print(f"Model Class Name: {class_name}")
            except Exception as e:
                print(f"Error reading config.json: {e}")
                
        if 'metadata.json' in z.namelist():
            try:
                meta_content = z.read('metadata.json').decode('utf-8')
                meta = json.loads(meta_content)
                keras_version = meta.get('keras_version', 'unknown')
                print(f"Keras Version from metadata.json: {keras_version}")
            except Exception as e:
                print(f"Error reading metadata.json: {e}")
    return True

def inspect_h5(filepath):
    print(f"\n--- Inspecting {filepath} as HDF5 ---")
    try:
        with h5py.File(filepath, 'r') as f:
            print("HDF5 Root Keys:", list(f.keys()))
            if 'model_config' in f.attrs:
                print("Found 'model_config' attribute in HDF5.")
                try:
                    config = json.loads(f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config'])
                    keras_version = config.get('keras_version', 'unknown')
                    print(f"Keras Version in model_config: {keras_version}")
                except Exception as e:
                    print(f"Error parsing model_config: {e}")
            else:
                print("No 'model_config' attribute found in HDF5 root attributes.")
    except Exception as e:
        print(f"Error opening as HDF5: {e}")

if __name__ == "__main__":
    keras_path = "eye_trained_model.keras"
    h5_path = "eye_trained_model.h5"
    
    if os.path.exists(keras_path):
        is_zip = inspect_keras_zip(keras_path)
        if not is_zip:
            inspect_h5(keras_path)
    else:
        print(f"{keras_path} not found.")
        
    if os.path.exists(h5_path):
        inspect_keras_zip(h5_path)
        inspect_h5(h5_path)
    else:
        print(f"{h5_path} not found.")
