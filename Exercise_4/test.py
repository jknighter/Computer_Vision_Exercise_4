import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from attrdict import AttrDict
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.initializers as initializers
import tensorflow.keras.preprocessing.image as kerasimage

# tf.debugging.set_log_device_placement(True)
#
# # Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print(c)
# print("status ok")

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# def create_model():
#   return tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
#   ])
#
# model = create_model()
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# log_dir="D:\\TFlogs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# model.fit(x=x_train,
#           y=y_train,
#           epochs=5,
#           validation_data=(x_test, y_test),
#           callbacks=[tensorboard_callback])

# m = np.empty((5,4,3))
# m = np.array([[[1,2],[3,3]],[[3,4],[4,4]],[[5,6],[7,5]]])
# # mt = m.T
# print(m.shape)
# a = np.max(m,axis=(0,1))
# print(a)

log_root= 'D:\\tmp\\tensorboard_logs'


def plot_multiple(images, titles=None, colormap='gray',
                  max_columns=np.inf, imwidth=4, imheight=4, share_axes=False):
    """Plot multiple images as subplots on a grid."""
    if titles is None:
        titles = [''] * len(images)
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * imwidth, n_rows * imheight),
        squeeze=False, sharex=share_axes, sharey=share_axes)

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis('off')

    if not isinstance(colormap, (list, tuple)):
        colormaps = [colormap] * n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()

dataset = tf.keras.datasets.cifar10
(im_train, y_train),(im_test, y_test) = dataset.load_data()
print("data load ok")
# Normalize to 0-1 range and subtract mean of training pixels
im_train = im_train / 255
im_test = im_test / 255

mean_training_pixel = np.mean(im_train, axis=(0,1,2))
x_train = im_train - mean_training_pixel
x_test = im_test - mean_training_pixel

image_shape = x_train[0].shape
labels = ['airplane','automobile','bird','cat',
          'deer','dog','frog','horse','ship','truck']


def train_model(model, batch_size=128, n_epochs=30, optimizer=optimizers.SGD,
                learning_rate=1e-2):
    opt = optimizer(lr=learning_rate)
    model.compile(
        optimizer=opt, loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = os.path.join(log_root, f'{model.name}_{timestamp}')
    tensorboard_callback = callbacks.TensorBoard(logdir, histogram_freq=1)
    # tensorboard = TensorBoard(log_dir=logdir.format(time()))
    model.fit(x=x_train, y=y_train, verbose=1, epochs=n_epochs,
              validation_data=(x_test, y_test), batch_size=batch_size,
              callbacks=[tensorboard_callback])

cnn = models.Sequential([
    # layers.Conv2D(filters=64, kernel_size=3, activation='relu',
    #               kernel_initializer='he_uniform', padding='same',
    #               input_shape=image_shape),
    # layers.MaxPooling2D(pool_size=2, strides=2),
    # layers.Conv2D(filters=64, kernel_size=3, activation='relu',
    #               kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten()],
    # layers.Dense(10, activation='softmax')],
    name='cnn')

train_model(cnn, optimizer=optimizers.Adam, learning_rate=1e-3, n_epochs=1)


def zero_padding(input, shape):
    pad_y, pad_x = np.subtract(shape[:2], input.shape[:2])
    res = np.pad(input, [(pad_y // 2, pad_y // 2), (pad_x // 2, pad_x // 2), (0, 0)], "constant")
    return res


def conv3x3_same(x, weights, biases):
    """Convolutional layer with filter size 3x3 and 'same' padding.
    `x` is a NumPy array of shape [height, width, n_features_in]
    `weights` has shape [3, 3, n_features_in, n_features_out]
    `biases` has shape [n_features_out]
    Return the output of the 3x3 conv (without activation)
    """
    # YOUR CODE HERE
    x_h, x_w = x.shape[:2]
    ker_h, ker_w, depth_in, depth_out = weights.shape
    result = np.zeros((x_h, x_w, depth_out))
    step_x = x_w - ker_w + 1
    step_y = x_h - ker_h + 1
    tmp = np.zeros((step_y, step_x, depth_out))

    for depth in range(depth_out):
        # w = weights.T[depth].T  # 3 * 3 * depth_in
        w = weights[:, :, :, depth].reshape(-1)
        for i in range(step_y):
            for j in range(step_x):
                patch = x[i:i + ker_h, j:j + ker_w].reshape(-1)  # 3 * 3 * depth_in
                # con1 = np.tensordot(w, patch, axes=([1,0],[0,1]))
                # con2 = np.tensordot(w, patch, axes=([1,0],[0,1])) + biases[depth]
                # tmp[i,j,depth] = np.sum(np.tensordot(w, patch, axes=([1,0],[0,1]))) + biases[depth]
                # w = weights.reshape(-1, depth_out)
                tmp[i, j, depth] = np.dot(w, patch) + biases[depth]

    # weights = weights.T #depth_out * depth_in * 3 * 3
    # for depth in range(depth_out):
    #     #wT = weights[depth] # depth_in * 3 * 3
    #     w = weights[depth].T # 3 * 3 * depth_in
    #     for i in range(step_y):
    #         for j in range(step_x):
    #             patch = x[i:i + ker_h, j:j + ker_w]  # 3 * 3 * depth_in
    #             for k in range(depth_in):
    #                 tmp[i,j,depth] += np.sum(np.multiply(patch[:,:,k], w[:,:,k]))
    #                 # con1 = np.tensordot(w, patch, axes=([1,0],[0,1]))
    #                 # con2 = np.tensordot(w, patch, axes=([1,0],[0,1])) + biases[depth]
    #                 # tmp[i,j,depth] = np.sum(np.tensordot(w, patch, axes=([1,0],[0,1]))) + biases[depth]
    #             tmp[i,j,depth] += biases[depth]
    result = zero_padding(tmp, x.shape)
    return result


def maxpool2x2(x):
    """Max pooling with pool size 2x2 and stride 2.
    `x` is a numpy array of shape [height, width, n_features]
    """
    # YOUR CODE HERE
    x_h, x_w, x_dep = x.shape
    win_size = 2
    stride = 2
    step_y = (x_h - win_size) // stride + 1
    step_x = (x_w - win_size) // stride + 1
    tmp = np.zeros((step_y, step_x, x_dep))
    for i in range(step_y):
        for j in range(step_x):
            tmp[i, j] = np.max(x[stride * i:stride * i + win_size, stride * j:stride * j + win_size], axis=(0, 1))
    # result = zero_padding(tmp, x.shape)
    result = tmp
    return result


def dense(x, weights, biases):
    # YOUR CODE HERE
    result = np.matmul(weights.T, x) + biases

    return result


def relu(x):
    # YOUR CODE HERE
    # print(x[np.where(x <= 0)])
    x = np.where(x <= 0, 0, x)
    return x


def softmax(x):
    # YOUR CODE HERE
    softmax = np.exp(x) / np.sum(np.exp(x))
    return softmax


def my_predict_cnn(x, W1, b1, W2, b2, W3, b3):
    x = conv3x3_same(x, W1, b1)
    x = relu(x)
    x = maxpool2x2(x)
    x = conv3x3_same(x, W2, b2)
    x = relu(x)
    x = maxpool2x2(x)
    x = x.reshape(-1)
    x = dense(x, W3, b3)
    x = softmax(x)
    return x

def test_conv(x, W1, b1, W2, b2):
    x = conv3x3_same(x, W1, b1)
    # x = relu(x)
    # x = maxpool2x2(x)
    # x = conv3x3_same(x, W2, b2)
    # x = relu(x)
    # x = maxpool2x2(x)
    x = x.reshape(-1)
    x = dense(x, W2, b2)
    x = softmax(x)
    return x

def test_maxpool(x, W1, b1):
    # x = conv3x3_same(x, W1, b1)
    # x = relu(x)
    # x = maxpool2x2(x)
    # x = conv3x3_same(x, W2, b2)
    # x = relu(x)
    x = maxpool2x2(x)
    x = x.reshape(-1)
    # x = dense(x, W1, b1)
    # x = softmax(x)
    return x


# m = np.empty((5,4,3))
# m = np.array([[[1,1,1],[2,2,2],[3,3,3],[4,4,4]],
#               [[5,5,5],[6,6,6],[7,7,7],[8,8,8]],
#               [[9,9,9],[0,0,0],[1,1,1],[2,2,2]],
#               [[3,3,3],[4,4,4],[5,5,5],[6,6,6]]])
# # mt = m.T
# print(m.shape)
# a = np.max(m,axis=(0,1))
# print(a)
# a = maxpool2x2(m)
# print(a)


# W1, b1 = cnn.layers[0].get_weights()
# W2, b2 = cnn.layers[2].get_weights()
# W3, b3 = cnn.layers[5].get_weights()

from random import randint

i_test = 12
# prob1 = []
# prob2 = []
# for i in range(10):
#     ran = randint(0, 100)
inp = x_test[i_test]
# my_prob = my_predict_cnn(inp, W1, b1, W2, b2, W3, b3)
my_pred = test_maxpool(inp, 0, 0)
# prob1.append(my_predict_cnn(inp, W1, b1, W2, b2, W3, b3))
# keras_prob = cnn.predict(inp[np.newaxis])[0]
keras_pred = cnn.predict(inp[np.newaxis])[0]
# prob2.append(cnn.predict(inp[np.newaxis])[0])
if np.mean((my_pred - keras_pred) ** 2) > 1e-10:
    print('Something isn\'t right! Keras gives different'
          'results than my_predict_cnn!')
    print("difference:")
    print(np.mean((my_pred - keras_pred) ** 2))
else:
    print('Congratulations, you got correct results!')

i_maxpred = np.argmax(my_pred)
# i_maxpred = np.argmax(keras_prob)
plot_multiple([im_test[i_test]],
              [f'Pred: {labels[i_maxpred]}, {my_pred[i_maxpred]:.1%}'],
              # [f'Pred: {labels[i_maxpred]}, {keras_prob[i_maxpred]:.1%}'],
              imheight=2)
plt.show()
# prob1 = np.array(prob1)
# prob2 = np.array(prob2)
# compare = np.append(prob1, prob2, axis=0)