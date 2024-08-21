import numpy as np
from tensorflow.keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()


z = np.zeros((60000, 2))
c = np.arange(60000)
print(z)
print(c)

def convert(y,num_classes):
    max_value = max(y)
    min_value = min(y)
    length = len(y)
    print(y)
    one_hot = np.zeros((length, num_classes))
    one_hot[np.arange(length), y] = 1
    print(one_hot)
    return one_hot

plt.plot(title="Model loss", figsize=(12, 8)).set(xlabel='Epoch', ylabel='loss')
plt.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')