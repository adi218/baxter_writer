import numpy as np
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile, join
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


def extract_files(files, digit):
    images_train = []
    images_val = []
    labels_train = []
    labels_val = []

    i = 0
    for f in files:
        i += 1
        if i > 4000:
            break
        if i < 500:
            img = cv2.imread(f)
            images_val.append(img)
        else:
            img = cv2.imread(f)
            images_train.append(img)
    # images = np.array(images)

    j = 0
    for i in range(len(files)):
        j += 1
        if j > 4000:
            break
        if j < 500:
            labels_val.append(digit)
        else:
            labels_train.append(digit)

    return images_train, images_val, labels_train, labels_val


all_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "+"]


path = "/Users/somanshusingh/PycharmProjects/digit_recog/data/=/"
files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
all_images_train, all_images_val, all_labels_train, all_labels_val = extract_files(files, 12)

for sym in all_symbols:
    print(sym)
    path = "/Users/somanshusingh/PycharmProjects/digit_recog/data/" + sym + "/"
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if sym == "-":
        images_train, images_val, labels_train, labels_val = extract_files(files, 10)
    elif sym == "+":
        images_train, images_val, labels_train, labels_val = extract_files(files, 11)
    else:
        images_train, images_val, labels_train, labels_val = extract_files(files, int(sym))

    # print(images.shape)
    # print(all_images.shape)
    all_images_train = np.concatenate((all_images_train, images_train), axis=0)
    all_images_val = np.concatenate((all_images_val, images_val), axis=0)
    all_labels_train = np.concatenate((all_labels_train, labels_train), axis=0)
    all_labels_val = np.concatenate((all_labels_val, labels_val), axis=0)


X_train = np.array(all_images_train)
y_train = np.array(all_labels_train)
X_validation = np.array(all_images_val)
y_validation = np.array(all_labels_val)

print(X_train.shape)
print(y_train.shape)
print(X_validation.shape)
print(y_validation.shape)

X_train = np.pad(X_train, ((0, 0), (2, 1), (2, 1), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 1), (2, 1), (0, 0)), 'constant')

print(X_train.shape)
print(X_validation.shape)


X_train_gray = []
X_valid_gray = []
X_test_gray = []
# print(X_train.shape)
for img in X_train:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_expanded = img[:, :, np.newaxis]
    X_train_gray.append(img_expanded)
X_train_gray = np.array(X_train_gray, dtype=np.float32)
for img in X_validation:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_expanded = img[:, :, np.newaxis]
    X_valid_gray.append(img_expanded)
X_valid_gray = np.array(X_valid_gray, dtype=np.float32)


X_train = X_train_gray
X_validation = X_valid_gray

print(X_train.shape)
print(X_validation.shape)

EPOCHS = 10
BATCH_SIZE = 128


def LeNet(x, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 48x48x1. Output = 44x44x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 44x44x6. Output = 22x22x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 18x18x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 18x18x16. Output = 9x9x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 1296.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 1296. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1296, 400), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
    # SOLUTION: Layer 4: Fully Connected. Input = 400. Output = 120.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(120))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

    fc3_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(84))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    # SOLUTION: Activation.
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc4_W = tf.Variable(tf.truncated_normal(shape=(84, 13), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(13))
    logits = tf.matmul(fc3, fc4_W) + fc4_b

    return logits


x = tf.placeholder(tf.float32, (None, 48, 48, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 13)

rate = 0.001

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

