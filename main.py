import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
from datetime import datetime


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


first = datetime.now()
# loading training set features
f = open("Datasets/train_set_features.pkl", "rb")
train_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=train_set_features2, axis=0)
train_set_features = train_set_features2[:, features_STDs > 52.3]

# changing the range of data between 0 and 1
train_set_features = np.divide(train_set_features, train_set_features.max())

# loading training set labels
f = open("Datasets/train_set_labels.pkl", "rb")
train_set_labels = pickle.load(f)
f.close()

# ------------
# loading test set features
f = open("Datasets/test_set_features.pkl", "rb")
test_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=test_set_features2, axis=0)
test_set_features = test_set_features2[:, features_STDs > 48]

# changing the range of data between 0 and 1
test_set_features = np.divide(test_set_features, test_set_features.max())

# loading test set labels
f = open("Datasets/test_set_labels.pkl", "rb")
test_set_labels = pickle.load(f)
f.close()

# ------------
# preparing our training and test sets - joining datasets and lables
train_set = []
test_set = []

for i in range(len(train_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(train_set_labels[i])] = 1
    label = label.reshape(4, 1)
    train_set.append((train_set_features[i].reshape(102, 1), label))

for i in range(len(test_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(test_set_labels[i])] = 1
    label = label.reshape(4, 1)
    test_set.append((test_set_features[i].reshape(102, 1), label))

# shuffle
random.shuffle(train_set)
random.shuffle(test_set)


W1 = np.random.normal(size=(150, 102))
W2 = np.random.normal(size=(60, 150))
W3 = np.random.normal(size=(4, 60))

# Initialize b = 0, for each layer.
b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))
learning_rate = 0.5  ######it didnt have good accuracy with learning_rate = 1 so i changed it to 0.15
number_of_epochs = 5
batch_size = 10
costs = []
for i in range(0, number_of_epochs):
    random.shuffle(train_set)
    batches = [train_set[x:x + batch_size] for x in range(0, 200, batch_size)]

    for batch in batches:
        grad_W1 = np.random.normal(size=(150, 102))
        grad_W2 = np.random.normal(size=(60, 150))
        grad_W3 = np.random.normal(size=(4, 60))

        # Initialize b = 0, for each layer.
        grad_b1 = np.zeros((150, 1))
        grad_b2 = np.zeros((60, 1))
        grad_b3 = np.zeros((4, 1))

        for image in batch:
            # compute the output (image is a0)
            a1 = sigmoid(W1 @ image[0] + b1)
            a2 = sigmoid(W2 @ a1 + b2)
            a3 = sigmoid(W3 @ a2 + b3)

            # last layer
            # weight
            for j in range(grad_W3.shape[0]):
                for k in range(grad_W3.shape[1]):
                    grad_W3[j, k] += 2 * (a3[j, 0] - image[1][j, 0]) * a3[j, 0] * (1 - a3[j, 0]) * a2[k, 0]

            # bias
            for j in range(grad_b3.shape[0]):
                grad_b3[j, 0] += 2 * (a3[j, 0] - image[1][j, 0]) * a3[j, 0] * (1 - a3[j, 0])

            # third layer
            # activation
            delta_3 = np.zeros((60, 1))
            for k in range(60):
                for j in range(4):
                    delta_3[k, 0] += 2 * (a3[j, 0] - image[1][j, 0]) * a3[j, 0] * (1 - a3[j, 0]) * W3[j, k]

            # weight
            for k in range(grad_W2.shape[0]):
                for m in range(grad_W2.shape[1]):
                    grad_W2[k, m] += delta_3[k, 0] * a2[k, 0] * (1 - a2[k, 0]) * a1[m, 0]

            # bias
            for k in range(grad_b2.shape[0]):
                grad_b2[k, 0] += delta_3[k, 0] * a2[k, 0] * (1 - a2[k, 0])

            # second layer
            # activation
            delta_2 = np.zeros((150, 1))
            for m in range(150):
                for k in range(60):
                    delta_2[m, 0] += delta_3[k, 0] * a2[k, 0] * (1 - a2[k, 0]) * W2[k, m]

            # weight
            for m in range(grad_W1.shape[0]):
                for v in range(grad_W1.shape[1]):
                    grad_W1[m, v] += delta_2[m, 0] * a1[m, 0] * (1 - a1[m, 0]) * image[0][v, 0]

            # bias
            for m in range(grad_b1.shape[0]):
                grad_b1[m, 0] += delta_2[m, 0] * a1[m, 0] * (1 - a1[m, 0])

            W1 = W1 - (learning_rate * (grad_W1 / batch_size))
            W2 = W2 - (learning_rate * (grad_W2 / batch_size))
            W3 = W3 - (learning_rate * (grad_W3 / batch_size))

            b1 = b1 - (learning_rate * (grad_b1 / batch_size))
            b2 = b2 - (learning_rate * (grad_b2 / batch_size))
            b3 = b3 - (learning_rate * (grad_b3 / batch_size))
    #  average for cost
    cost = 0
    for train_data in train_set[:200]:
        a0 = train_data[0]
        a1 = sigmoid(W1 @ a0 + b1)
        a2 = sigmoid(W2 @ a1 + b2)
        a3 = sigmoid(W3 @ a2 + b3)
        for j in range(4):
            cost += np.power((a3[j, 0] - train_data[1][j, 0]), 2)
    cost /= 200
    costs.append(cost)

count_correct = 0
for train_data in train_set[0:200]:
    a0 = train_data[0]
    a1 = sigmoid(W1 @ a0 + b1)
    a2 = sigmoid(W2 @ a1 + b2)
    a3 = sigmoid(W3 @ a2 + b3)  # for one of the fruits each time
    max_prediction_index = a3.argmax()
    max_prediction = a3[max_prediction_index]
    max_real_index = train_data[1].argmax()
    max_real = train_data[1][max_real_index]
    prediction = np.where(a3 == max_prediction[0])
    real = np.where(train_data[1] == max_real[0])
    if real == prediction:
        count_correct += 1
print("accuracy: ", count_correct / 200)
epoch_size = [x for x in range(5)]
plt.plot(epoch_size, costs)
plt.show()
last = datetime.now()
print("duration time: ", last-first)
