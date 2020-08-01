import os
import datetime

##### Misc.
def exit():
    os._exit(0)

def GetTimeString(m = -1):
    if m==0:
        s1 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        s1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    return s1


def GetDateString():
    s1 = datetime.datetime.now().strftime("%Y%m%d")
    return s1

def MakeDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


##### ML Part

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_to_one_hot(Y, C):
    Y = np.array(int(Y))
    Y = np.eye(C)[Y.reshape(-1)]
    return Y[0]


def plot_confusion_matrix(actual, predicted, classes, title='Confusion Matrix'):
    conf_matrix = pd.crosstab(actual, predicted)  # confusion_matrix(actual, predicted)

    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.viridis)  # , cmap=plt.cm.Greens
    plt.title(title, size=12)
    plt.colorbar(fraction=0.05, pad=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    '''
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
        horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")
    '''
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)
    plt.tight_layout()


import seaborn as sns
def plot_feature_distribution(data_train, data_test, features, bw1=0.5, rows=2, cols=4, figsize=(16, 8)):
    i = 0
    sns.set_style('whitegrid')
    f1 = plt.figure()

    label1 = 'train'
    label2 = 'test'
    if data_train.shape[1] > 1:
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        for feature in features:
            if i < features.size:
                plt.subplot(rows, cols, i + 1)
                print(data_train.shape)  # (1665000, 1)
                # (1665000, 2)

                sns.kdeplot(data_train[i], bw=bw1, label=label1)
                sns.kdeplot(data_test[i], bw=bw1, label=label2)

                plt.xlabel(feature, fontsize=9)
                locs, labels = plt.xticks()
                plt.tick_params(axis='x', which='major', labelsize=8)
                plt.tick_params(axis='y', which='major', labelsize=8)
            i += 1
    else:
        # print(features.shape)
        fig = plt.figure()
        sns.kdeplot(data_train[:, 0], bw=bw1, label=label1)
        sns.kdeplot(data_test[:, 0], bw=bw1, label=label2)
        plt.xlabel(features[0], fontsize=9)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    # plt.show()
    return fig  # f1;


def plot_feature_class_distribution(data, classes, features, bw1=0.5, rows=2, cols=3, figsize=(16, 16 / 2)):
    i = 0
    sns.set_style('whitegrid')
    # plt.figure()
    # ncols = round(datadim/2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # print(data[class_name[0]].shape[2])
    # classes = data.keys()
    # print(classes)
    # print(len(classes))

    for feature in features:
        ax = plt.subplot(rows, cols, i + 1)
        for clas in classes:
            # print(clas)
            # print(data[clas].shape)

            datafeature = data[clas][:, :, i].reshape(-1)
            # print(datafeature.shape)

            sns.kdeplot(datafeature, bw=bw1, label=clas)

        # ax[i].xlabel(feature, fontsize=9)

        # ax.set_xlabel('aa')
        ax.set_xlim([-3, 3])
        ax.set_xlabel(feature)
        i += 1
        # print()

    return fig
    '''
    for feature in features:
        plt.subplot(2,4,i+1)
        sns.kdeplot(data_train[i], bw=0.5,label=label1)
        sns.kdeplot(data_test[i], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
        i += 1
    plt.show()
    '''