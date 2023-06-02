from keras.models import *

# Load YAML file
with open('models/cfg/SRyolo_noFocus.json', 'r') as json_file:
    loaded_model_yaml = json_file.read()

# Load model architecture from YAML file
loaded_model = model_from_json(loaded_model_yaml)
model_from_yaml

# Save model weights to H5 file
loaded_model.save_weights('SRyolol_noFocus.h5')
