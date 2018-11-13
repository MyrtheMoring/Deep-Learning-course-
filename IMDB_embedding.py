import re
import numpy as np
from numpy import array
import collections
from collections import Counter
import pandas as pd
import time
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
import string
from string import punctuation

# Packages for modeling
import tensorflow as tf
from keras import models, layers, regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from keras.layers import Dense, Activation, Embedding,Flatten
from sklearn.metrics import confusion_matrix, classification_report

""" Lab2: Word Embedding model
    Data: IMDB movie reviews
    Type: classification

    Data: 13/11/2018
    Name: Myrthe Moring """

# ______________________________________________________________________________
# Functions:
""" Cleaning Reviews:
    1. remove punctuation from each token
    2. remove remaining tokens that are not alphabetic
    3. filter out stop words """
def cleaning_review(review):
    review_words = (re.sub("[^a-zA-Z@]", " ", review)).lower().split()
    review_tokens = [t for t in review_words if (t.isalpha() and not re.match("^[@]", t))]
    return " ".join(review_tokens)

""" Split data:
    1. train
    2. test
    3. validation """
def split_data(r, labels, frac, max_len):
    f = np.zeros((len(r), max_len), dtype=int)
    for i, row in enumerate(r):
        f[i, -len(row):] = np.array(row)[:max_len]

    i1 = int(len(f) * frac)
    train_x, val_x = f[:i1], f[i1:]
    train_y, val_y = labels[:i1], labels[i1:]

    i2 = int(len(val_x) * frac/2)
    val_x, test_x = val_x[:i2], val_x[i2:]
    val_y, test_y = val_y[:i2], val_y[i2:]

    return [train_x, train_y, val_x, val_y, test_x, test_y]

""" Deep model:
    1. compile
    2. fit
    3. return the history """
def deep_model(model, X_train, y_train, X_valid, y_valid, lr, opt):

    model.compile(optimizer=opt, loss='categorical_crossentropy'
                  , metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=nr_epochs,
                        batch_size=size_batch,
                        validation_data=(X_valid, y_valid),
                        verbose=0)
    return history

""" Evaluate the model by plotting the accuracy and loss curves. """
def evaluate_model(history, met):
    plt.plot(history.history[met], color="blue")
    plt.plot(history.history['val_' + met], color="red")
    plt.title(met + ' plot of model')
    plt.ylabel(met)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    return plt

if __name__ == '__main__':

    print("Starting:", time.ctime())

    " Read the data and clean the reviews "
    dataset_location = "dataset_imdb.csv"
    IMDB = pd.read_csv(dataset_location, encoding = "ISO-8859-1")
    IMDB['clean_review'] = IMDB['SentimentText'].apply(lambda l: cleaning_review(l))
    texts = ' '.join(IMDB['clean_review'])
    words = texts.split()
    counts = Counter(words)

    "Define the number of words in vocabulary"
    number_words = 800

    "Make a vocabulary "
    v = sorted(counts, key=counts.get, reverse=True)[:number_words]
    vocab = {word: w for w, word in enumerate(v, 1)}
    r = []
    for each in IMDB['clean_review']:
        r.append([vocab[word] for word in each.split() if word in vocab])

    " Define max length (number of words) of review "
    review_len = Counter([len(x) for x in r])
    max_review_len = max(review_len)

    " Create list of labels for the sentiment "
    labels = np.array(IMDB['Sentiment'])
    review_id = [i for i, review in enumerate(r) if len(review) > 0]
    labels = labels[review_id]
    r = [rev for rev in r if len(rev) > 0]

    " Split the data in a train, test and validation set "
    [train_x, train_y, val_x, val_y, test_x, test_y] = split_data(r, labels, 0.7, max_review_len)

    " Make the labels suitable for the model "
    train_y_c = np_utils.to_categorical(train_y, 3)
    val_y_c = np_utils.to_categorical(val_y, 3)
    test_y_c = np_utils.to_categorical(test_y, 3)

    print("X and y values")
    print("Train set: ", train_x.shape, train_y.shape,
          "\nValidation set: ", val_x.shape,val_y.shape,
          "\nTest set: ", test_x.shape, test_y.shape)
# ______________________________________________________________________________
    # Model:
    " Parameters "
    embedding = 16
    nr_epochs = 64
    size_batch = 512

    "Defining the model "
    emb_model = models.Sequential()
    emb_model.add(layers.Embedding(1000, embedding, input_length=max_review_len))
    emb_model.add(layers.Flatten())
    emb_model.add(layers.Dense(3, activation='softmax'))
    emb_model.summary()

    " Learning rate and optimizer "
    lr = 0.01
    opt = SGD(lr=lr, momentum=0.95)

    print("Time starts at : ", time.ctime())
    emb_history = deep_model(emb_model, train_x, train_y_c, val_x, val_y_c, lr, opt)
    print("Time ends at : ", time.ctime())

# ______________________________________________________________________________
    # Results:
    score, accuracy = emb_model.evaluate(test_x, test_y_c,
                                batch_size=size_batch,
                                verbose=0)

    test_pred = emb_model.predict_classes(test_x, verbose=0)
    class_report = classification_report(test_y, test_pred)
    C_matrix = confusion_matrix(test_y, test_pred)

    print("Results: \n")
    print('Test accuracy= ', accuracy)
    print()
    print('Test score= ', score)
    print()
    print('Classification Report\n', class_report)
    print()
    print('Confusion Matrix\n', C_matrix)

    " Print the accuracy and loss plots "
    acc = evaluate_model(emb_history, "acc")
    loss = evaluate_model(emb_history, "loss")

    # acc.savefig('acc_plot.png')
    # loss.savefig('loss_plot.png')
