import numpy as np
import cv2
import tensorflow as tf
import extract_char as im
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt


def LeNet(x, keep_prob):
    mu = 0
    sigma = 0.1

    # Layer 1
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc0 = flatten(conv2)

    # Layer 3
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1296, 400), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    # Layer 4
    fc2_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(120))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

    #  Layer 5
    fc3_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(84))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)

    #  Layer 6
    fc4_W = tf.Variable(tf.truncated_normal(shape=(84, 13), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(13))
    logits = tf.matmul(fc3, fc4_W) + fc4_b

    return logits


camera_file = "camera_image.jpg"
img = cv2.imread(camera_file)
# print(img)
# plt.imshow(img)
# plt.show()
num_images = im.image_segmentation(img)

x = tf.placeholder(tf.float32, (None, 48, 48, 1))
keep_prob = tf.placeholder(tf.float32)

logits = LeNet(x, keep_prob)
operation = 1
answer = 0
print(num_images)
for i in range(num_images):
    filename = str(i + 1) + ".png"
    img = cv2.imread(filename)

    orig_color = (50, 50, 50)
    replacement_color = (255, 255, 255)
    img[(img >= orig_color).all(axis=-1)] = replacement_color

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[:, :, np.newaxis]

    img = np.array([img])
    img = np.pad(img, ((0, 0), (2, 1), (2, 1), (0, 0)), 'constant')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        output = sess.run(logits, feed_dict={x: img, keep_prob: 1})
        symbol = sess.run(tf.argmax(output, axis=1))

    if symbol[0] == 10:
        operation = 0
    elif symbol[0] == 11:
        operation = 1
    elif symbol[0] == 12:
        continue
    else:
        if operation == 0:
            answer -= symbol[0]
        else:
            answer += symbol[0]

with open('answer.txt', 'w') as f:
    f.write(str(answer))
