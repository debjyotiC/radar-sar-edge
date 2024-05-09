import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import matplotlib.pyplot as plt

load_data = np.load("data/npz_files/umbc_outdoor_processed.npz", allow_pickle=True)

classes = 4

ground_mask = np.ones((9, 256))
ground_mask[:, :10] = 0

x_data = load_data['out_x'] * ground_mask
y_data = load_data['out_y']

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)

train_ratio = 0.70
test_ratio = 1 - train_ratio

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Create the ANN model
model = tf.keras.Sequential([
    layers.Flatten(input_shape=x_data[0].shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(classes, activation='softmax')
])

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# this controls the batch size
BATCH_SIZE = 40
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)

history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Training Accuracy: {round(np.average(acc), 3)}")
print(f"Validation Accuracy: {round(np.average(val_acc), 3)}")

predicted_labels = model.predict(x_test)
actual_labels = y_test

# np.savez('test-results/test-gray-results.npz', out_x=predicted_labels, out_y=actual_labels)

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
