import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist


def convert_to_binary(y):
    return np.array([1 if label == 5 else 0 for label in y])


def convert_from_one_hot(y):
    y_origin = np.argmax(y, axis=1)
    return y_origin


def model(input_shape):
    input_frame = tf.keras.Input(shape=input_shape)
    f1 = tf.keras.layers.Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.03))(input_frame)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(f1)
    mode = tf.keras.Model(inputs=input_frame, outputs=output_layer)
    return mode


(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(train_X.shape[0], -1).astype('float32')


test_X = test_X.reshape(test_X.shape[0], -1).astype('float32')

train_X = train_X / 255

test_X = test_X / 255


train_y = convert_to_binary(train_y)


test_y = convert_to_binary(test_y)

nn_model = model(train_X.shape[1:])


nn_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

nn_model.summary()
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(buffer_size=10000).batch(train_X.shape[0]
                                                                                                        )

test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(test_X.shape[0])

history = nn_model.fit(train_dataset, epochs=100, validation_data=test_dataset)

df_loss_acc = pd.DataFrame(history.history)
print(df_loss_acc)
df_loss = df_loss_acc[['loss', 'val_loss']].copy()
df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
df_acc = df_loss_acc[['accuracy', 'val_accuracy']].copy()
df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)

print(df_loss)
print(df_acc)


# Plot loss
plt.plot(df_loss.index, df_loss['train'], label='Training Loss')
plt.plot(df_loss.index, df_loss['validation'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(df_acc.index, df_acc['train'], label='Training Accuracy')
plt.plot(df_acc.index, df_acc['validation'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
