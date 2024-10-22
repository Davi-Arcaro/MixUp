# https://keras.io/examples/vision/mixup/ Project following this documentation
import os
os.environ["Keras_Backend"] = "tensorflow"

import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers
import tensorflow as tf

#TF imports related to tf.data preprocessing
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow.random import gamma as tf_random_gamma # type: ignore

#Preparing the dataset
#Using FashionMNIST dataset

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
y_train = keras.ops.one_hot(y_train, 10)

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
y_test = keras.ops.one_hot(y_test, 10)

#Defining the hyperparameters
auto = tf_data.AUTOTUNE
batch_size = 64
epochs = 10

#Converting the data into TensorFlow Dataset objects

#Put aside a few samples to create the validation set
val_samples = 2000
x_val, y_val = x_train[:val_samples], y_train[:val_samples]
new_x_train, new_y_train = x_train[val_samples:], y_train[val_samples:]

train_ds_one = (tf_data.Dataset.from_tensor_slices((new_x_train, new_y_train)).shuffle(batch_size * 100).batch(batch_size))
train_ds_two = (tf_data.Dataset.from_tensor_slices((new_x_train, new_y_train)).shuffle(batch_size * 100).batch(batch_size))

#Combining two shuffled datasets from the same training data because it'll be mixing up the images and their corresponding labels
train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))
val_ds = tf_data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

#Define the mixup technique function
#Create new virtual datasets using the training data from the same dataset and apply a lambda value within the [0,1] range sampled from a Beta distribution.

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf_random_gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf_random_gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def mix_up(ds_one, ds_two, alpha=0.2):
    #Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = keras.ops.shape(images_one)[0]

    #convert images to float32
    images_one = tf.cast(images_one, tf.float32)
    images_two = tf.cast(images_two, tf.float32)

    #sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = keras.ops.reshape(l, (batch_size, 1, 1, 1))
    y_l = keras.ops.reshape(l, (batch_size, 1))

    #perform mixup on both images and labels by combining a pair of images/labels (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1-x_l)
    labels = labels_one * y_l + labels_two * (1-y_l)
    return (images, labels)

#Visualizing the new augmented dataset
#first create the new dataset using our 'mix_up' utility
train_ds_mu = train_ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=auto,)

#preview 9 samples from the dataset
sample_images, sample_labels = next(iter(train_ds_mu))
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    plt.axis('off')
    # plt.show()


#Model Building
def get_training_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ]
    )
    return model

#for the sake of reproducibility, we serialize the initial random weights of our shallow network
initial_model = get_training_model()
initial_model.save_weights('initial_weights.weights.h5')

#Train the model with the mixed up dataset
model = get_training_model()
model.load_weights('initial_weights.weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_ds_mu, validation_data=val_ds, epochs=epochs)
_, test_acc = model.evaluate(test_ds)
print('Test accuracy: {:.2f}%'.format(test_acc * 100))


#NOTE
#You can train the model without the mixedup dataset, just comment out the last section of code and uncomment this one to do so
# model = get_training_model()
# model.load_weights('initial_weights.weights.h5')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #Notice that we are NOT using the mixedup dataset here
# model.fit(train_ds_one, validation_data=val_ds, epochs=epochs)
# _, test_acc = model.evalueate(test_ds)
# print('Test accuracy: {:.2f}%'.format(test_acc * 100))