# Lab 5: CNN Model for Binary Image Classification

## Using TensorFlow/Keras and Cats vs Dogs Dataset

### 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** for binary image classification.

The model classifies images into two categories:

* 🐱 Cat
* 🐶 Dog

For this project, the **CIFAR-10 dataset** is used. Only the Cat and Dog classes are selected from the original 10-class dataset, making it a binary classification problem.

The complete implementation is provided as a **Google Colab `.ipynb` notebook**.

---

## 2. Objectives

The main objectives of this lab are:

1. Understand the basic architecture of a Convolutional Neural Network.
2. Implement a CNN using TensorFlow/Keras.
3. Perform image preprocessing.
4. Apply data augmentation techniques.
5. Train a CNN for binary image classification.
6. Evaluate the trained model.
7. Analyze prediction results using classification metrics and a confusion matrix.

---

## 3. Dataset

### CIFAR-10 Dataset

CIFAR-10 contains 60,000 color images belonging to 10 different classes.

For this project, only two classes are selected:

| Original CIFAR-10 Label | Class |
| ----------------------: | ----- |
|                       3 | Cat   |
|                       5 | Dog   |

The original labels are converted into binary labels:

```text
Cat → 0
Dog → 1
```

Each image has a size of:

```text
32 × 32 × 3
```

where:

* 32 = image height
* 32 = image width
* 3 = RGB color channels

The dataset is automatically downloaded by TensorFlow, so no manual dataset upload is required.

---

## 4. Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Google Colab**

---

## 5. CNN Architecture

The CNN used in this project contains the following major layers:

```text
Input Image
     ↓
Data Augmentation
     ↓
Conv2D (32 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (64 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (128 filters)
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense (128 neurons)
     ↓
Dropout (0.5)
     ↓
Dense (1 neuron, Sigmoid)
     ↓
Cat / Dog
```

### Convolutional Layers

The convolutional layers extract important visual features such as:

* Edges
* Shapes
* Textures
* Patterns
* Object features

### Max Pooling

Max pooling reduces the spatial dimensions of feature maps and helps reduce computational complexity.

### Flatten Layer

The extracted feature maps are converted into a one-dimensional vector before passing them to the fully connected layers.

### Dense Layer

The dense layer learns higher-level relationships between extracted features.

### Dropout

A dropout rate of `0.5` is used to reduce overfitting during training.

### Sigmoid Output

Since this is a binary classification problem, the final layer uses:

```python
Dense(1, activation="sigmoid")
```

The output represents the probability of the image being a dog.

---

## 6. Image Preprocessing

The original image pixel values range from:

```text
0 to 255
```

They are normalized to:

```text
0 to 1
```

using:

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

Normalization helps the neural network train more efficiently.

---

## 7. Data Augmentation

Data augmentation is applied to improve model generalization and reduce overfitting.

The following techniques are used:

### Random Horizontal Flip

```python
layers.RandomFlip("horizontal")
```

### Random Rotation

```python
layers.RandomRotation(0.1)
```

### Random Zoom

```python
layers.RandomZoom(0.1)
```

These transformations generate slightly different versions of training images while preserving their class.

---

## 8. Model Compilation

The CNN is compiled using the **Adam optimizer**:

```python
optimizer="adam"
```

Since the problem is binary classification, the loss function is:

```python
loss="binary_crossentropy"
```

The main evaluation metric is:

```python
metrics=["accuracy"]
```

---

## 9. Model Training

The model is trained using:

```text
Epochs      = 15
Batch Size  = 64
Validation  = 20% of training data
```

Training is performed using:

```python
model.fit()
```

During training, both training and validation accuracy/loss are monitored.

---

## 10. Model Evaluation

After training, the model is evaluated using the test dataset.

The following metrics are generated:

* Test Loss
* Test Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The test accuracy is displayed automatically when the notebook is executed.

---

## 11. Classification Report

The classification report provides detailed performance information for both classes.

It contains:

### Precision

Measures how many predicted samples of a class are actually correct.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Support

The number of actual samples belonging to each class.

---

## 12. Confusion Matrix

The confusion matrix shows how the model's predictions compare with the actual labels.

The matrix contains:

```text
                 Predicted
                 Cat    Dog

Actual Cat       TN     FP
Actual Dog       FN     TP
```

This helps identify which class the CNN is confusing more often.

---

## 13. Prediction Visualization

The notebook displays several test images along with:

```text
Actual: Cat/Dog
Predicted: Cat/Dog
```

This provides a visual interpretation of the CNN's classification performance.

The notebook also performs a prediction on an individual test image and displays the predicted class probability.

---

## 14. Model Saving

After training, the trained model is saved as:

```text
cats_vs_dogs_cnn.keras
```

The model can later be loaded using:

```python
model = tf.keras.models.load_model("cats_vs_dogs_cnn.keras")
```

---

## 15. Project Files

The project contains:

```text
Lab_5_CNN_Cats_vs_Dogs/
│
├── Lab_5_CNN_Cats_vs_Dogs.ipynb
├── cats_vs_dogs_cnn.keras
└── README.md
```

The `.ipynb` file contains the complete Google Colab implementation.

---

## 16. How to Run the Project

### Step 1 — Open Google Colab

Open Google Colab in your browser.

### Step 2 — Upload Notebook

Upload:

```text
Lab_5_CNN_Cats_vs_Dogs.ipynb
```

### Step 3 — Run the Notebook

Run each cell from top to bottom.

The notebook automatically:

1. Imports required libraries.
2. Downloads CIFAR-10.
3. Selects Cat and Dog images.
4. Preprocesses the images.
5. Applies data augmentation.
6. Creates the CNN.
7. Trains the model.
8. Evaluates the model.
9. Generates graphs.
10. Generates a classification report.
11. Creates a confusion matrix.
12. Displays predictions.
13. Saves the trained model.

---

## 17. Expected Output

After running the notebook, the following outputs will be generated:

* Dataset information
* Sample Cat and Dog images
* CNN model summary
* Training progress
* Test accuracy
* Training vs validation accuracy graph
* Training vs validation loss graph
* Classification report
* Confusion matrix
* Prediction visualization
* Individual image prediction
* Saved CNN model

---

## 18. Learning Outcomes

After completing this experiment, the following concepts are demonstrated:

* Understanding of CNN architecture
* Image classification using deep learning
* TensorFlow/Keras model development
* Image normalization
* Data augmentation
* Convolution and pooling
* Binary classification
* Model training and validation
* Performance evaluation
* Confusion matrix interpretation
* Prediction analysis

---

## 19. Conclusion

A Convolutional Neural Network was successfully implemented using **TensorFlow/Keras** for binary image classification.

The CIFAR-10 dataset was converted into a two-class dataset containing **Cats and Dogs**. Image preprocessing and data augmentation were applied before training.

The CNN was trained and evaluated using a separate test dataset. Accuracy, loss, classification metrics, confusion matrix, and prediction visualizations were used to assess the performance of the model.

This experiment demonstrates how CNNs can automatically learn visual features from images and use those features to perform image classification.

---

## 20. Lab Information

**Lab:** 5
**Topic:** CNN Model
**Framework:** TensorFlow/Keras
**Dataset:** CIFAR-10 — Cat and Dog classes
**Task:** Binary Image Classification
**Environment:** Google Colab / Python

**Submitted By:** ____________________
**Roll Number:** ____________________
**Course:** ____________________
**Faculty:** Ramesh Chandra Poonia
**School:** SCHOOL OF SCIENCES (NCR)
