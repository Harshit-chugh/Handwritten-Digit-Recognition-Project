import tensorflow as tf
from tensorflow.keras import layers, models
objects =  tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = objects.load_data()

train_images  = train_images.reshape((60000,28,28,1)).astype('float32') / 255
test_images = test_images.reshape((10000,28,28,1)).astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

model = models.Sequential()

# Adding 1st Convolution Layer
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Adding 2nd Convolution Layer
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))

# Output layer
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=25, batch_size = 100, validation_split= 0.2)

model.save('model.h5')