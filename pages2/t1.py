# from keras.models import load_model
from keras.models import load_model

# Path to your old .h5 model
legacy_model_path = "Stock-Prediction/models/model0.h5"
converted_model_path = "Stock-Prediction/models/model0.keras"

# Load legacy model
legacy_model = load_model(legacy_model_path, compile=False)

# Save in new format
legacy_model.save(converted_model_path)
print(f"✅ Model converted and saved as {converted_model_path}")
