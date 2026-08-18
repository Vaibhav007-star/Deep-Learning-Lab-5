# Lab 5: CNN Model for Binary Image Classification
# TensorFlow/Keras - Cats vs Dogs
# Google Colab Notebook

# ============================================================
# CELL 1 — Install / Import Libraries
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow version:", tf.__version__)


# ============================================================
# CELL 2 — Download and Prepare Dataset
# ============================================================
# We use CIFAR-10 and keep only two classes:
# Cat = 3
# Dog = 5
#
# This creates a binary image-classification dataset.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

CAT = 3
DOG = 5

train_mask = (y_train == CAT) | (y_train == DOG)
test_mask = (y_test == CAT) | (y_test == DOG)

x_train = x_train[train_mask]
y_train = y_train[train_mask]

x_test = x_test[test_mask]
y_test = y_test[test_mask]

# Cat = 0, Dog = 1
y_train = (y_train == DOG).astype(np.float32)
y_test = (y_test == DOG).astype(np.float32)

print("Training images:", len(x_train))
print("Testing images :", len(x_test))


# ============================================================
# CELL 3 — Image Preprocessing
# ============================================================
# Convert pixel values from 0–255 to 0–1.

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Image shape:", x_train.shape[1:])
print("Preprocessing completed.")


# ============================================================
# CELL 4 — Display Sample Images
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i])
    label = "Dog" if y_train[i] == 1 else "Cat"
    plt.title(label)
    plt.axis("off")

plt.suptitle("Sample Images — Cats vs Dogs")
plt.tight_layout()
plt.show()


# ============================================================
# CELL 5 — Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
], name="data_augmentation")

print("Data augmentation created.")


# ============================================================
# CELL 6 — Build CNN Model
# ============================================================

model = models.Sequential([
    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

model.summary()


# ============================================================
# CELL 7 — Compile CNN
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully.")


# ============================================================
# CELL 8 — Train CNN
# ============================================================

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# CELL 9 — Evaluate Model
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# CELL 10 — Accuracy Graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# CELL 11 — Loss Graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# CELL 12 — Predictions
# ============================================================

probabilities = model.predict(x_test, verbose=0)
y_pred = (probabilities >= 0.5).astype(int).flatten()

print("Predictions generated successfully.")


# ============================================================
# CELL 13 — Classification Report
# ============================================================

print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_test.astype(int),
    y_pred,
    target_names=["Cat", "Dog"]
))


# ============================================================
# CELL 14 — Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test.astype(int), y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.show()


# ============================================================
# CELL 15 — Show Predictions
# ============================================================

plt.figure(figsize=(12, 9))

for i in range(12):
    plt.subplot(3, 4, i + 1)

    plt.imshow(x_test[i])

    actual = "Dog" if y_test[i] == 1 else "Cat"
    predicted = "Dog" if y_pred[i] == 1 else "Cat"

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis("off")

plt.suptitle("CNN Prediction Results")
plt.tight_layout()
plt.show()


# ============================================================
# CELL 16 — Save Trained Model
# ============================================================

model.save("cats_vs_dogs_cnn.keras")

print("Model saved as cats_vs_dogs_cnn.keras")


# ============================================================
# CELL 17 — Predict One New Test Image
# ============================================================

index = 25

image = x_test[index]
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)[0][0]

predicted_class = "Dog" if prediction >= 0.5 else "Cat"
actual_class = "Dog" if y_test[index] == 1 else "Cat"

plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.title(
    f"Actual: {actual_class}\n"
    f"Predicted: {predicted_class}\n"
    f"Dog Probability: {prediction:.2%}"
)
plt.axis("off")
plt.show()


# ============================================================
# LAB 5 — FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("LAB 5 COMPLETED")
print("=" * 60)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("Dataset: CIFAR-10 (Cat and Dog classes)")
print("Model: Convolutional Neural Network (CNN)")
print("Framework: TensorFlow / Keras")
print("Techniques: Preprocessing + Data Augmentation")
