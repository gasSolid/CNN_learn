import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

for i in [tf, np]:
    print(i.__name__, ": ", i.__version__, sep="")

mnist = tf.keras.datasets.mnist
(trainImage, trainLabel), (testImage, testLabel) = mnist.load_data()

# for i in [trainImage,trainLabel,testImage,testLabel]:
#    print(i.shape)

trainImage = tf.reshape(trainImage, (60000, 28, 28, 1))
testImage = tf.reshape(testImage, (10000, 28, 28, 1))

#for i in [trainImage, trainLabel, testImage, testLabel]:
#    print(i.shape)

net = tf.keras.models.Sequential([
    # 卷积层1
    tf.keras.layers.Conv2D(filters=6, kernel_size=(
        5, 5), activation="relu", input_shape=(28, 28, 1), padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    # 卷积层2
    tf.keras.layers.Conv2D(filters=16, kernel_size=(
        5, 5), activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # 卷积层3
    tf.keras.layers.Conv2D(filters=32, kernel_size=(
        5, 5), activation="relu", padding="same"),
    # tf.keras.layers.MaxPool2D(pool_size=2,strides=2),

    tf.keras.layers.Flatten(),

    # 全连接层1
    tf.keras.layers.Dense(200, activation="relu"),

    # 全连接层2
    tf.keras.layers.Dense(10, activation="softmax")
])
net.summary()

net.compile(optimizer='adam',
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = net.fit(trainImage, trainLabel, epochs=5, validation_split=0.1)

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()