# Bharat-Intern-Task-3
Number Recognition

TASK 3: Number Recognition

By Aluri Srihari Goud

About Dataset
MNIST is a commonly used dataset in machine learning and computer vision research, which consists of a set of 70,000 images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is split into two sets: a training set of 60,000 images and a test set of 10,000 images. The training set is used to train a machine learning model, while the test set is used to evaluate the model's performance.

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import unique , argmax

# TensorFlow already contain MNIST data set which can be loaded using Keras
import tensorflow as tf # installing tenserflow
from tensorflow import keras
# To Load the MNIST dataset from the Keras API provided by TensorFlow.
mnist = tf.keras.datasets.mnist
The Above Code Reflects that the Dataset Contains :

An array of 60,000 images, each represented as a 28x28 NumPy array, with pixel values ranging from 0 to 255.
An array of 60,000 labels, each representing the correct digit (0-9) for the 1.
An array of 10,000 images, each represented as a 28x28 NumPy array, with pixel values ranging from 0 to 255.
An array of 10,000 labels, each representing the correct digit (0-9) for the 3.
Dividing the data into train and test data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
(60000, 28, 28)
(60000,)
(10000, 28, 28)
(10000,)
print(x_train)
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 ...

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]]
print(x_test)
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 ...

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]]
# Reshaping the input Data which is used as a input in CNN in Tenserflow
# CNN takes the input Data in 4D Format with the shape (num_samples, image_height, image_width, num_channels)
# Here (num_channels) is set to 1 which means input image is Grayscale.

x_train = x_train.reshape((x_train.shape[0] , x_train.shape[1] , x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0] , x_test.shape[1] , x_test.shape[2],1))
print(x_train.shape)
print(x_test.shape)
print(x_train.dtype)
print(x_test.dtype)
(60000, 28, 28, 1)
(10000, 28, 28, 1)
uint8
uint8
# Normalizing Pixel Values

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print(x_train.dtype)
print(x_test.dtype)
float32
float32
# Visulaizing Subsets of images in MNIST Dataset along with coressponding labels.

fig=plt.figure(figsize=(5,3))
for i in range(20):
    ax =fig.add_subplot(2,10,i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]), cmap='Blues')
    ax.set_title(y_train[i])

    ![image](https://github.com/srihari06/Bharat-Intern-Task-3/assets/119677889/8f812580-3c30-43b6-a45a-838de81b5734)

    # showing shape of single image
img_shape= x_train.shape[1:]
img_shape
(28, 28, 1)
BUILDING NEURAL NETWORK THAT CAN READ HANDWRITTEN DIGITS.
# Creating aSequential Model in Keras
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
model.summary()
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten_1 (Flatten)         (None, 784)               0         
                                                                 
 dense_2 (Dense)             (None, 128)               100480    
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_3 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
This summary shows that the model has four layers:

A Flatten layer that flattens the input images to a 1D array of length 784.
A Dense layer with 128 units and ReLU activation.
A Dropout layer that randomly sets 20% of the input units to 0 during training.
A second Dense layer with 10 units and no activation function.
The summary also shows the number of trainable parameters in each layer, as well as the total number of trainable parameters in the model. In this case, the model has a total of 101,770 trainable parameters.

# Displaying Neural Network Model
from tensorflow.keras.utils import plot_model
plot_model(model, 'model.jpg', show_shapes = True)

![image](https://github.com/srihari06/Bharat-Intern-Task-3/assets/119677889/e7bbc14f-6cf9-40e8-a076-e0a886c5f65f)

# Making Prediction on Model
prediction = model(x_train[:1]).numpy()
prediction
array([[ 0.08819406,  0.02334788,  0.05840464,  0.06903427,  0.5252824 ,
        -0.04078927, -0.22387438, -0.43524188, -0.27941954, -0.03243962]],
      dtype=float32)
# Applying Softmax() Function to prediction array
# This convert an output vector of real numbers into a probability distribution over predicted classes
tf.nn.softmax(prediction).numpy()
array([[0.10846904, 0.10165843, 0.10528546, 0.10641058, 0.16793098,
        0.09534305, 0.07939189, 0.06426588, 0.07510229, 0.09614246]],
      dtype=float32)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], prediction).numpy()
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
Model fitting
# Training the Model
model.fit(x_train, y_train, epochs=5)
Epoch 1/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.2960 - accuracy: 0.9140
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.1439 - accuracy: 0.9571
Epoch 3/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1089 - accuracy: 0.9672
Epoch 4/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0890 - accuracy: 0.9727
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0757 - accuracy: 0.9764
<keras.callbacks.History at 0x7f41a1286830>
# Evaluating the Model
model.evaluate(x_test, y_test, verbose=2)
313/313 - 1s - loss: 0.0752 - accuracy: 0.9770 - 545ms/epoch - 2ms/step
[0.07518018782138824, 0.9769999980926514]
# Creating a new sequential model which includes both previously trained model and softmax layer.
probability_model = tf.keras.Sequential([ model,tf.keras.layers.Softmax() ])
probability_model(x_test[:5])
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[1.5023636e-08, 2.8046278e-08, 8.6892478e-06, 2.1114422e-05,
        8.8602803e-10, 4.8365621e-08, 6.1451641e-13, 9.9996769e-01,
        2.8051858e-07, 2.1160774e-06],
       [3.8681939e-08, 4.1496165e-05, 9.9991894e-01, 1.7614640e-05,
        4.3373732e-15, 1.6626309e-05, 3.2137373e-06, 9.0283954e-13,
        2.0866528e-06, 7.1583733e-13],
       [3.7049006e-07, 9.9759489e-01, 1.6454932e-04, 5.1491074e-06,
        3.9168852e-05, 6.7521983e-06, 4.3748798e-05, 1.8154874e-03,
        3.2646983e-04, 3.4668781e-06],
       [9.9989104e-01, 1.0030488e-07, 3.4515320e-05, 4.4909646e-07,
        1.5330463e-05, 2.1342439e-06, 2.7114160e-05, 2.8521930e-05,
        7.9536186e-08, 5.7907613e-07],
       [4.1292051e-06, 6.2117507e-09, 5.1304491e-05, 1.7004332e-07,
        9.9647850e-01, 4.2222570e-07, 3.9773659e-06, 8.5955893e-05,
        2.9344408e-06, 3.3726362e-03]], dtype=float32)>
# Displaying a Grayscale Image
img = x_train[12]
plt.imshow(np.squeeze(img) ,cmap='gray')
plt.show()

![image](https://github.com/srihari06/Bharat-Intern-Task-3/assets/119677889/8fdd8101-4572-4017-b81c-b94088986a34)

# Predicting the Result
img= img.reshape(1, img.shape[0],img.shape[1],img.shape[2])
p= model.predict([img])
print("predicted : {}".format(argmax(p)))
1/1 [==============================] - 0s 55ms/step
predicted : 3



