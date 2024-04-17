import numpy as np
from os import listdir
from os.path import isdir, join
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.load("data/npz_files/umbc_outdoor_processed.npz", allow_pickle=True)

x_data = data['out_x']
y_data = data['out_y']

print(x_data.shape)

dataset_path = 'data/csv_files/umbc'

all_targets = [target for target in listdir(dataset_path) if isdir(join(dataset_path, target))]
classes = len(all_targets)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6)

x_train = tf.expand_dims(x_train, axis=-1)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((9, 256, 1), input_shape=x_train.shape[1:]),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])

# model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), metrics=['acc'])

# this controls the batch size
BATCH_SIZE = 10
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

history = model.fit(train_dataset, epochs=1000, validation_data=validation_dataset)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Training Accuracy: {round(np.average(acc), 3)}")
print(f"Validation Accuracy: {round(np.average(val_acc), 3)}")

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1)

# plot loss
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')
# plot accuracy
axs[1].plot(epochs, acc, '-', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.show()
