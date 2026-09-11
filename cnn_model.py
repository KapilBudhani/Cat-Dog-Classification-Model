import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMG_SIZE = 128
BATCH_SIZE = 32
DATASET_PATH = "PetImages"


def get_valid_files():
    class_names = ["Cat", "Dog"]
    supported_formats = {"JPEG", "PNG", "BMP", "GIF"}

    files = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_PATH, class_name)

        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)

            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                continue

            try:
                if os.path.getsize(file_path) == 0:
                    continue

                with Image.open(file_path) as image:
                    image_format = image.format

                    if image_format not in supported_formats:
                        continue

                    image.load()
                    image.convert("RGB")

                files.append(file_path)
                labels.append(label)

            except Exception:
                continue

    return np.array(files), np.array(labels, dtype=np.float32), class_names


def load_image(file_path, label):
    def process_image(path):
        path = path.numpy().decode("utf-8")

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image = np.asarray(image, dtype=np.float32) / 255.0

        return image

    image = tf.py_function(
        func=process_image,
        inp=[file_path],
        Tout=tf.float32
    )

    image.set_shape((IMG_SIZE, IMG_SIZE, 3))

    return image, label


def load_dataset():
    files, labels, class_names = get_valid_files()

    indices = np.arange(len(files))

    np.random.seed(123)
    np.random.shuffle(indices)

    files = files[indices]
    labels = labels[indices]

    split_index = int(0.8 * len(files))

    train_files = files[:split_index]
    train_labels = labels[:split_index]

    validation_files = files[split_index:]
    validation_labels = labels[split_index:]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_files, train_labels)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_files, validation_labels)
    )

    train_dataset = train_dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = (
        train_dataset
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Valid images: {len(files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(validation_files)}")

    return train_dataset, validation_dataset, class_names


def build_model():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.5),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model