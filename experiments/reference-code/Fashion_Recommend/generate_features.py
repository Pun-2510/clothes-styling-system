import os
import pickle as pkl
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm


# =========================
# CONFIG
# =========================
IMAGE_DIR = "archive/images"
N = 1000

FEATURES_FILE = "Images_features.pkl"
FILENAMES_FILE = "filenames.pkl"


# =========================
# LOAD MODEL
# =========================
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    GlobalMaxPool2D()
])


# =========================
# GET IMAGE FILES
# =========================
filenames = []

for file in os.listdir(IMAGE_DIR):
    file_path = os.path.join(IMAGE_DIR, file)

    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        filenames.append(file_path)

filenames.sort()

print("Total images:", len(filenames))

# Chỉ lấy N ảnh
filenames = filenames[:N]

print("Images used:", len(filenames))


# =========================
# EXTRACT FEATURES
# =========================
def extract_features(image_path):
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_expand_dim = np.expand_dims(img_array, axis=0)

    img_preprocess = preprocess_input(img_expand_dim)

    result = model.predict(
        img_preprocess,
        verbose=0
    ).flatten()

    norm_result = result / norm(result)

    return norm_result


# =========================
# PROCESS IMAGES
# =========================
image_features = []

for i, file in enumerate(filenames):
    try:
        feature = extract_features(file)
        image_features.append(feature)

        print(f"[{i + 1}/{len(filenames)}] {file}")

    except Exception as e:
        print(f"Error: {file}")
        print(e)


# =========================
# SAVE
# =========================
pkl.dump(
    image_features,
    open(FEATURES_FILE, "wb")
)

pkl.dump(
    filenames,
    open(FILENAMES_FILE, "wb")
)


print("\nDone!")
print("Features:", FEATURES_FILE)
print("Filenames:", FILENAMES_FILE)
print("Number of features:", len(image_features))