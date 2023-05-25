# MNIST Digit Recognition
![image](https://github.com/SP85691/Digit_Recognition/assets/86033489/769b27ea-e9d6-4779-b9bb-e322b9cf5da4)
# Project Summary
This project focuses on building and training a deep learning model for digit classification using the MNIST dataset. The MNIST dataset is a collection of handwritten digit images along with their corresponding labels, making it a popular benchmark dataset in the field of machine learning.

**The project follows the following steps:**

- **Step 1:** Importing the MNIST Dataset: The MNIST dataset is imported using the TensorFlow library, providing access to the training and testing subsets of handwritten digit images.
  ```
  import tensorflow as tf
  mnist = tf.keras.datasets.mnist
  ```

- **Step 2:** Loading and Splitting the MNIST Dataset: The dataset is loaded and split into training and testing subsets using the mnist.load_data() function. The images and labels are stored in variables X_train, Y_train, X_test, and Y_test.
  ```
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  ```
- **Step 3:** Normalizing the MNIST Dataset: The pixel values of the MNIST dataset are normalized using the tf.keras.utils.normalize() function, which scales the values to a range of 0 to 1.
  ```
  X_train = tf.keras.utils.normalize(X_train, axis=1)
  X_test = tf.keras.utils.normalize(X_test, axis=1)
  plt.imshow(X_train[0], cmap=plt.cm.binary)
  ```
- **Step 4:** Resizing Images for Convolutional Operations: The images are resized to include the channel dimension required for applying convolutional operations in a CNN. This is done using the np.array().reshape() function.
  ```
  import numpy as np
  IMG_SIZE = 28

  X_trainr = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  X_testr = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  ```
- **Step 5:** Creating a Deep Neural Network for MNIST Classification: A deep neural network model is created using the Sequential API from TensorFlow. The model consists of convolutional layers, activation layers, max pooling layers, a flattening layer, and fully connected layers.<br>
  ![image](https://github.com/SP85691/Digit_Recognition/assets/86033489/39578078-2c0d-43af-9577-128fd92feaa0)
- **Step 6:** Compiling the CNN Model: The model is compiled using the model.compile() function, specifying the loss function, optimizer, and metrics for training and evaluation.
  ```
  model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
  model.fit(X_trainr, Y_train, epochs=5, validation_split=0.3)
  ```
- **Step 7:** Training and Evaluating the Model: The model is trained on the training dataset and evaluated on the testing dataset using the model.fit() function. The training process involves adjusting the model's parameters to minimize the loss and improve accuracy.
  ```
  test_loss, test_acc = model.evaluate(X_testr, Y_test)
  print("Test Loss on 10000 Test Samples: ", test_loss)
  print("Validation Accuracy on 10000 test samples: ", test_acc)
  ```
- **Step 8:** Predicting Digit Labels: The trained model is used to predict the labels for new images or unseen data.

# **Licenses, Credits, Requirements, Datasets, etc.**
## **Licenses**
- The code and documentation in this project are released under the <a href="https://opensource.org/license/mit/" target="_blank"><u>**`MIT License`**</u></a>.

## **Credits**
This project is created by `[Surya Pratap]` and is based on the MNIST dataset and TensorFlow library.

## **Requirements**
```
Python 3.6+
TensorFlow 2.x
Keras
Matplotlib
Numpy
Pandas
Opencv-Python
```
## **Datasets**
- The MNIST dataset is used in this project. It consists of a collection of handwritten digit images along with their corresponding labels. The dataset is divided into a training set of 60,000 images and a test set of 10,000 images. Each image is a 28x28 grayscale image.

## **References**
<a href="http://yann.lecun.com/exdb/mnist/" target="_blank"><u>**MNIST Dataset**</u></a><br>
<a href="https://www.tensorflow.org/" target="_blank"><u>**TensorFlow Documentation**</u></a><br>
<a href="https://docs.python.org/3/" target="_blank"><u>**Python Documentation**</u></a><br>

## **Getting Started**
### **1. Clone the repository:**
```
git clone https://github.com/SP85691/Digit_Recognition.git
```
### **2. Create the Virtual Environment:**
```
python -m venv venv
```
### **3. Install the required dependencies:**
```
pip install -r requirements.txt
```
### **4. Run the Jupyter File:**
```
python main.py
```
