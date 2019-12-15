import sys
import ast
import numpy as np
import csv
import random
from sklearn.utils import shuffle
import math
import sys

def delta(emotion_vecs, v, covar, input_vec, N):
    output = np.linalg.slogdet(np.add(covar, N*np.matmul(v, v.T)))
    return output[0]*output[1]

def newsum(emotion_vecs, has_input, input_vec):
    current_sum = emotion_vecs[0]
    for vec in emotion_vecs[1:]:
        current_sum = np.add(current_sum, vec)
    if has_input:
        current_sum = np.add(current_sum, input_vec)
    return current_sum

def calculatek(emotion_vecs, v, covar, input_vec, N):
    return np.matmul(np.matmul(np.matmul((np.linalg.inv(np.add(covar, N*np.matmul(v, v.T)))), v), v.T), np.linalg.inv(covar))

def log_likelihood(emotion_vecs, mean, v, covar, input_vec):
    # subtract m from all vectors
    new_emotion_vecs = []
    for vec in emotion_vecs:
        new_emotion_vecs.append(np.subtract(vec, mean))
    new_emotion_vecs = np.array(new_emotion_vecs)
    input_vec = np.subtract(input_vec, mean)
    
    first_term = delta(new_emotion_vecs, v, covar, input_vec, len(new_emotion_vecs) + 1)
    second_term = np.matmul(np.matmul(newsum(new_emotion_vecs, True, input_vec).T, calculatek(new_emotion_vecs, v, covar, input_vec, len(new_emotion_vecs) + 1)), newsum(new_emotion_vecs, True, input_vec))
    third_term = delta(new_emotion_vecs, v, covar, input_vec, len(new_emotion_vecs))
    fourth_term = np.matmul(np.matmul(newsum(new_emotion_vecs, False, input_vec).T, calculatek(new_emotion_vecs, v, covar, input_vec, len(new_emotion_vecs))), newsum(new_emotion_vecs, False, input_vec))
    fifth_term = delta(new_emotion_vecs, v, covar, input_vec, 1)
    sixth_term = 1/2*np.matmul(np.matmul(input_vec.T, calculatek(new_emotion_vecs, v, covar, input_vec, 1)), input_vec)

    ratio = -1*first_term - second_term + third_term + fourth_term + fifth_term + sixth_term

    return ratio


meld = open("meld_xvectors.csv")
reader = csv.reader(meld, delimiter=',')

train = []
train_labels = []
test = []
test_labels = []
ad = 0
h = 0
s = 0
n = 0
fs = 0
ad_val = []
h_val = []
s_val = []
n_val = []
fs_val = []

line = 0
for row in reader:
    if line == 0:
        line += 1
    else:
        random_number = random.random()
        if random_number < 0.8 and len(train) < 10966:
            train.append(ast.literal_eval(row[4]))
            if row[2] == "anger/disgust":
                train_labels.append(0)
                ad += 1
                ad_val.append(ast.literal_eval(row[4]))
            elif row[2] == "happiness":
                train_labels.append(1)
                h += 1
                h_val.append(ast.literal_eval(row[4]))
            elif row[2] == "sadness":
                train_labels.append(2)
                s += 1
                s_val.append(ast.literal_eval(row[4]))
            elif row[2] == "neutral":
                train_labels.append(3)
                n += 1
                n_val.append(ast.literal_eval(row[4]))
            elif row[2] == "fear/surprise":
                train_labels.append(4)
                fs += 1
                fs_val.append(ast.literal_eval(row[4]))
        else:
            test.append(ast.literal_eval(row[4]))
            if row[2] == "anger/disgust":
                test_labels.append(0)
            elif row[2] == "happiness":
                test_labels.append(1)
            elif row[2] == "sadness":
                test_labels.append(2)
            elif row[2] == "neutral":
                test_labels.append(3)
            elif row[2] == "fear/surprise":
                test_labels.append(4)
        line += 1

train = np.array(train)
train_labels = np.array(train_labels)
test = np.array(test)
test_labels = np.array(test_labels)

x = shuffle(train, train_labels)
new_train = x[0]
new_train_labels = x[1]
y = shuffle(test, test_labels)
new_test = y[0]
new_test_labels = y[1]

new_file_lda = open("output_lda")
# parameters of LDA
matrix = []
for row in new_file_lda:
    count = 0
    for x in row.split(" "):
        if x != '\n' and x and x != ']\n' and x != '[\n':
            if count < 512:
                matrix.append(float(x))
            count += 1
matrix = np.array(matrix)
matrix = np.reshape(matrix, (512, 200))

# put the train and test sets through LDA
ad_val = np.array(ad_val)
h_val = np.array(h_val)
s_val = np.array(s_val)
n_val = np.array(n_val)
fs_val = np.array(fs_val)
ad_val = np.matmul(ad_val, matrix)
h_val = np.matmul(h_val, matrix)
s_val = np.matmul(s_val, matrix)
n_val = np.matmul(n_val, matrix)
fs_val = np.matmul(fs_val, matrix)
new_test = np.matmul(new_test, matrix)

new_file = open("output_plda")

# parameters of PLDA
mean = None
V = []
covar = None

count = 0
list_string = ''
for row in new_file:
    if count == 0:
        mean = list(map(float, (row[7:].split(" ")[2:-1])))
    elif count < 202:
        list_string += row
    elif count == 202:
        for x in list_string.split(" "):
            if x != '\n' and x and x != ']\n' and x != '[\n':
                V.append(float(x))
        covar = list(map(float, (row.split(" ")[2:-1])))
    count += 1

mean = np.array(mean, dtype=np.float64)
V = np.array(V, dtype=np.float64)
V = np.reshape(V, (200, 200))
covar = np.diag(covar)

correct = 0
total = 0
# loop through tests
for i in range(len(new_test)):
    # actual label
    actual = new_test_labels[i]
    scores = []
    # do a multisession scoring for each class
    # http://faculty.iitmandi.ac.in/~padman/papers/padman_ivecAvgPLDA_DSP_2014.pdf
    for j in range(5):
        if j == 0:
            scores.append(log_likelihood(ad_val, mean, V, covar, new_test[i]))
        elif j == 1:
            scores.append(log_likelihood(h_val, mean, V, covar, new_test[i]))
        elif j == 2:
            scores.append(log_likelihood(s_val, mean, V, covar, new_test[i]))
        elif j == 3:
            scores.append(log_likelihood(n_val, mean, V, covar, new_test[i]))
        elif j == 4:
            scores.append(log_likelihood(fs_val, mean, V, covar, new_test[i]))
    if actual == scores.index(max(scores)):
        correct += 1
    total += 1

print(correct/total)