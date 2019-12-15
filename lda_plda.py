import csv
import ast
import random
import numpy as np
from lda import LDA
from sklearn.utils import shuffle
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from classifier import Classifier
from model import Model

meld = open("meld_xvectors.csv")
reader = csv.reader(meld, delimiter=',')

train = []
train_labels = []
test = []
test_labels = []

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
            elif row[2] == "happiness":
                train_labels.append(1)
            elif row[2] == "sadness":
                train_labels.append(2)
            elif row[2] == "neutral":
                train_labels.append(3)
            elif row[2] == "fear/surprise":
                train_labels.append(4)
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

lda = LDA()
lda.fit(new_train, new_train_labels)
# print('train score', lda.score(new_train, new_train_labels))
# print('test score', lda.score(new_test, new_test_labels))
new_x = lda.transform(new_train, n_components=200)
testing_new = lda.transform(new_test, n_components=200)

# plda
second_classifier = Classifier()
# 5 classes
second_classifier.fit_model(new_x, new_train_labels)
predictions, log_p_predictions = second_classifier.predict(testing_new)
print('Accuracy: {}'.format((new_test_labels == predictions).mean()))
print(predictions)
print(new_test_labels)
predictions, log_p_predictions = second_classifier.predict(new_x)
print('Accuracy: {}'.format((new_train_labels == predictions).mean()))
print(second_classifier.posterior_params[category]['mean'])
print(second_classifier.posterior_params[category]['cov_diag'])
print(second_classifier.Psi)