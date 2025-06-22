# 🧠 Handwritten Digit Classification using Artificial Neural Networks (ANN)

This project demonstrates a basic implementation of an Artificial Neural Network (ANN) to classify handwritten digits from the popular **MNIST dataset**. The goal of this project is to understand how to build, train, and evaluate a neural network using **Keras** and **TensorFlow** for a multiclass classification problem.

---

## 🔍 Dataset
* Contains grayscale images of handwritten digits:

  * 60,000 training images
  * 10,000 test images
  * Each image is 28×28 pixels (784 features)
* **Target classes**: Digits from `0` to `9`

---

## 📌 Objective

The primary goal of this project is **learning**:

* How to train a basic  network for multiclass classification

---

## 🧠 Technologies Used

* Python
* NumPy
* Matplotlib (for data visualization)
* TensorFlow / Keras (for building and training the ANN)

---

## 🧪 Steps Followed

1. **Data Loading & Visualization**

   * Load the MNIST dataset using `tensorflow.keras.datasets`
   * Visualize sample images to understand the data

2. **Preprocessing**

   * Flatten 2D images (28x28) into 1D vectors (784,)
   * Normalize pixel values to the range `[0, 1]`

3. **Building the Neural Network**

   * Used `Sequential` API from Keras
   * Hidden layer with 128 neurons and ReLU activation
   * Hidden layer with 32 neurons and ReLU activation
   * Output layer with 10 neurons (for 10 digits) and softmax activation

4. **Training & Evaluation**

   * Compiled with `sparse_categorical_crossentropy` loss and `adam` optimizer
   * Evaluated using test accuracy and confusion matrix

---

## 🧾 Model Summary

```python
model= Sequential()

model.add(Flatten(input_shape=(28,28))) 
model.add(Dense(128,activation='relu',))
model.add(Dense(32,activation='relu',))
model.add(Dense(10,activation='softmax')
```

---

## 📈 Results

* The model achieves around **97% accuracy** on the test set
* Includes loss/accuracy curves.


