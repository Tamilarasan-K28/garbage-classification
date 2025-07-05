import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model("model/waste_classifier_model.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

img_path = "data/test_image.jpg"

if not os.path.exists(img_path):
    print(f"❌ Image not found: {img_path}")
    exit()

# Load and preprocess
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
label = class_names[np.argmax(prediction)]

print("✅ Predicted Class:", label)
