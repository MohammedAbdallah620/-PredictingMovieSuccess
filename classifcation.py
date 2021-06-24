import cv2
import numpy as np
import os
from random import shuffle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from DataCleaner import *
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def adaBost(X_train,X_test,y_train,y_test,with_pc, pca):
    if with_pc==1:
        filename='adapost_treepca' + str(pca) + '.sav'
    else:
        filename = 'AdaTree'
    if os.path.exists(filename):
        adaBostTree = pickle.load(open(filename, 'rb'))
    else:
        adaBostTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=200).fit(X_train,
                                                                                                     y_train)
        pickle.dump(adaBostTree, open(filename, 'wb'))

    y_pred = adaBostTree.predict(X_test)
    print("Accuracy in tree Adabost:  ", metrics.accuracy_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)
    print('conf matrix in adabost tree')
    print(conf_mat)



################################

    if with_pc == 1:
        filename = 'AdaSVM_PCA' + str(pca) + '.sav'
    else:
        filename = 'AdaSVM'

    if os.path.exists(filename):
        adaBostsvm = pickle.load(open(filename, 'rb'))
    else:
        svc = SVC(probability=True, kernel='linear')
        adaBostsvm = AdaBoostClassifier(n_estimators=2, base_estimator=svc, learning_rate=1).fit(X_train, y_train)
        pickle.dump(adaBostTree, open(filename, 'wb'))

    y_pred = adaBostsvm.predict(X_test)
    print("Accuracy in svm Adabost:", metrics.accuracy_score(y_test, y_pred))

    conf_mat = confusion_matrix(y_test, y_pred)
    print('conf matrix in adabost svm ')
    print(conf_mat)


def SVM(X_train, X_test, y_train, y_test, withpca, pca):
    filenames = []
    if withpca == 1:
        filenames = ['SVM1pca' + str(pca) + '.sav', 'SVM2pca' + str(pca) + '.sav', 'SVM3pca' + str(pca) + '.sav',
                     'SVM4pca' + str(pca) + '.sav', 'SVM5pca' + str(pca) + '.sav', 'SVM7pca' + str(pca) + '.sav',
                     'SVM6pca' + str(pca) + '.sav', 'SVM8pca' + str(pca) + '.sav', 'SVM9pca' + str(pca) + '.sav']
    else:
        filenames = ['SVM1.sav', 'SVM2.sav', 'SVM3.sav', 'SVM4.sav', 'SVM5.sav', 'SVM7.sav', 'SVM6.sav',
                     'SVM8.sav', 'SVM9.sav']
    c = [0.1, 1, 1000]
    ker = ['linear', 'poly', 'rbf']
    training_time, testing_time, max_accuracy = 0, 0, 0
    max_model=None
    for i in range(len(c)):
        for j in range(len(ker)):
            k = i * len(c) + j
            filename = filenames[k]
            if os.path.exists(filename):
                svm = pickle.load(open(filename, 'rb'))
            else:
                t1 = time.time()
                svm = OneVsRestClassifier(SVC(kernel=ker[j], C=c[i])).fit(X_train, y_train)
                t2 = time.time()
                training_time = max(training_time, t2 - t1)
                pickle.dump(svm, open(filename, 'wb'))
            t1 = time.time()
            accuracy = svm.score(X_test, y_test) * 100
            t2 = time.time()
            testing_time = max(testing_time, t2 - t1)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                max_model=svm
            max_accuracy = max(accuracy, max_accuracy)
            # print('One VS Rest SVM accuracy with kernel={} and c={} is : {}%'.format(ker[j], c[i],  accuracy))
    y_pred = max_model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("conf_mat of SVM ")
    print(conf_mat)
    return training_time, testing_time, max_accuracy


def KNN(X_train, X_test, y_train, y_test, withpca, pca):
    weight = ['uniform', 'distance']
    max_model=None
    filenames = []
    for i in range(40):
        if withpca == 1:
            filenames.append('KNN' + str(i) + 'pca' + str(pca) + '.sav')
        else:
            filenames.append('KNN' + str(i) + '.sav')
    training_time, testing_time, max_accuracy = 0, 0, 0
    for j in range(0, len(weight)):
        for i in tqdm(range(1, 20)):
            k = j * 19 + i
            filename = filenames[k]
            if os.path.exists(filename):
                knn = pickle.load(open(filename, 'rb'))
            else:
                t1 = time.time()
                knn = KNeighborsClassifier(n_neighbors=i, weights=weight[j])
                knn.fit(X_train, y_train)
                t2 = time.time()
                training_time = max(training_time, t2 - t1)
                pickle.dump(knn, open(filename, 'wb'))
            t1 = time.time()
            # print("#####################" , X_train.shape , X_test.shape)
            # print(np.reshape(X_train.shape , -1))
            pred_i = knn.predict(X_test)
            accuracy = np.mean(pred_i == y_test) * 100
            t2 = time.time()
            testing_time = max(testing_time, t2 - t1)
            if accuracy>max_accuracy:
                max_model=knn
                max_accuracy=accuracy
            # print('KNN accuracy: with n_neighbors={} and  weights={} is: {}%'.format(i ,weight[j] ,accuracy))
    y_pred = max_model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("conf_mat of knn ")
    print(conf_mat)
    return training_time, testing_time, max_accuracy


def Logistic_Regression(X_train, X_test, y_train, y_test, withpca, pca):
    c = [1, 1000, 1000000]
    max_model=None
    filenames = []
    if withpca == 1:
        filenames = ['Logistic_Regressionpca1' + str(pca) + '.sav', 'Logistic_Regressionpca2' + str(pca) + '.sav',
                     'Logistic_Regressionpca3' + str(pca) + '.sav']
    else:
        filenames = ['Logistic_Regression1.sav', 'Logistic_Regression2.sav', 'Logistic_Regression3.sav']
    training_time, testing_time, max_accuracy = 0, 0, 0
    for i in range(len(c)):
        filename = filenames[i]
        if os.path.exists(filename):
            logistic_regression_model = pickle.load(open(filename, 'rb'))
        else:
            t1 = time.time()
            logistic_regression_model = LogisticRegression(C=c[i]).fit(X_train, y_train)
            t2 = time.time()
            training_time = max(training_time, t2 - t1)
            pickle.dump(logistic_regression_model, open(filename, 'wb'))
        t1 = time.time()
        accuracy = logistic_regression_model.score(X_test, y_test) * 100
        t2 = time.time()
        testing_time = max(testing_time, t2 - t1)
        if accuracy>max_accuracy:
            max_model=logistic_regression_model
            max_accuracy=accuracy
        # print('Logistic Regression accuracy: ' + str(accuracy))
    y_pred = max_model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("conf_mat of Logistic Regression")
    print(conf_mat)
    return training_time, testing_time, max_accuracy


def Decision_Tree(X_train, X_test, y_train, y_test, withpca, pca):
    max_feature = ['sqrt', 'log2', None]
    depth = [5, 10, 20, 50, None]
    max_model=None

    filenames = []
    if withpca == 1:
        filenames = ['DT1pca' + str(pca) + '.sav', 'DT2pca' + str(pca) + '.sav', 'DT3pca' + str(pca) + '.sav',
                     'DT4pca' + str(pca) + '.sav', 'DT5pca' + str(pca) + '.sav', 'DT6pca' + str(pca) + '.sav',
                     'DT7pca' + str(pca) + '.sav', 'DT8pca' + str(pca) + '.sav', 'DT9pca' + str(pca) + '.sav',
                     'DT10pca' + str(pca) + '.sav', 'DT11pca' + str(pca) + '.sav', 'DT12pca' + str(pca) + '.sav',
                     'DT13pca' + str(pca) + '.sav', 'DT14pca' + str(pca) + '.sav', 'DT15pca' + str(pca) + '.sav']
    else:
        filenames = ['DT1.sav', 'DT2.sav', 'DT3.sav', 'DT4.sav', 'DT5.sav', 'DT6.sav', 'DT7.sav', 'DT8.sav',
                     'DT9.sav', 'DT10.sav', 'DT11.sav', 'DT12.sav', 'DT13.sav', 'DT14.sav', 'DT15.sav']
    training_time, testing_time, max_accuracy = 0, 0, 0
    for i in range(len(max_feature)):
        for j in range(len(depth)):
            k = i * len(depth) + j
            filename = filenames[k]
            if os.path.exists(filename):
                decision_tree_model = pickle.load(open(filename, 'rb'))
            else:
                t1 = time.time()
                decision_tree_model = DecisionTreeClassifier(max_features=max_feature[i], max_depth=depth[j]).fit(
                    X_train, y_train)
                t2 = time.time()
                training_time = max(training_time, t2 - t1)
                pickle.dump(decision_tree_model, open(filename, 'wb'))
            t1 = time.time()
            accuracy = decision_tree_model.score(X_test, y_test) * 100
            t2 = time.time()
            testing_time = max(testing_time, t2 - t1)
            if accuracy > max_accuracy:
                max_model = decision_tree_model
                max_accuracy = accuracy
            # print('Logistic Regression accuracy: ' + str(accuracy))
        y_pred = max_model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        print("conf_mat of decision_tree_model")
        print(conf_mat)

    return training_time, testing_time, max_accuracy


def classification_withoutPCA(X_train, X_test, y_train, y_test, withpca):
    adaBost(X_train, X_test, y_train, y_test, withpca, 0)
    svm_triningTime, svm_testingTime, svm_accuracy = SVM(X_train, X_test, y_train, y_test, withpca, 0)
    knn_triningTime, knn_testingTime, knn_accuracy = KNN(X_train, X_test, y_train, y_test, withpca, 0)
    LR_triningTime, LR_testingTime, LR_accuracy = Logistic_Regression(X_train, X_test, y_train, y_test, withpca, 0)
    DT_triningTime, DT_testingTime, DT_accuracy = Decision_Tree(X_train, X_test, y_train, y_test, withpca, 0)
    print("fot svm, training time, testing time, accuracy")
    print(svm_triningTime, svm_testingTime, svm_accuracy)
    print("fot knn, training time, testing time, accuracy")

    print(knn_triningTime, knn_testingTime, knn_accuracy)
    print("fot LR, training time, testing time, accuracy")

    print(LR_triningTime, LR_testingTime, LR_accuracy)
    print("fot DT, training time, testing time, accuracy")

    print(DT_triningTime, DT_testingTime, DT_accuracy)

    #########bar graphs
    model_name = ('SVM', 'KNN', 'Logistic Regression', 'Decision Tree')
    y_pos = np.arange(len(model_name))
    training_time = [svm_triningTime, knn_triningTime, LR_triningTime, DT_triningTime]
    testing_time = [svm_testingTime, knn_testingTime, LR_testingTime, DT_testingTime]
    accuracy = [svm_accuracy, knn_accuracy, LR_accuracy, DT_accuracy]

    plt.bar(y_pos, training_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Training Time')
    plt.show()

    plt.bar(y_pos, testing_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Testing Time')
    plt.show()

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('accuracy')
    plt.title('Acuuracy')
    plt.show()


def Pca(train, test, component):
    pca = PCA(n_components=component)
    principalComponents = pca.fit_transform(train)
    principalComponents2 = pca.transform(test)
    print(principalComponents.shape, principalComponents2.shape)
    # print(pca.explained_variance_ratio_)
    X_train = pd.DataFrame(data=principalComponents
                           , columns=['pca'] * component)
    X_test = pd.DataFrame(data=principalComponents2
                          , columns=['pca'] * component)
    return X_train, X_test


def classification_with_PCA(X_train, X_test, y_train, y_test, withpca):
    '''plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('movies Dataset Explained Variance')
    plt.show()'''
    components = [150, 450, 950]
    svm_TriningTime, svm_TestingTime, svm_Accuracy = 0, 0, 0
    knn_TriningTime, knn_TestingTime, knn_Accuracy = 0, 0, 0
    LR_TriningTime, LR_TestingTime, LR_Accuracy = 0, 0, 0
    DT_TriningTime, DT_TestingTime, DT_Accuracy = 0, 0, 0
    for i in range(len(components)):
        print('pca wirh {} components'.format(components[i]))
        X_train_pca, X_test_pca = Pca(X_train, X_test, components[i])
        adaBost(X_train_pca, X_test_pca, y_train, y_test, withpca, i)
        svm_triningTime, svm_testingTime, svm_accuracy = SVM(X_train_pca, X_test_pca, y_train, y_test, withpca, i)
        knn_triningTime, knn_testingTime, knn_accuracy = KNN(X_train_pca, X_test_pca, y_train, y_test, withpca, i)
        LR_triningTime, LR_testingTime, LR_accuracy = Logistic_Regression(X_train_pca, X_test_pca, y_train, y_test, withpca, i)
        DT_triningTime, DT_testingTime, DT_accuracy = Decision_Tree(X_train_pca, X_test_pca, y_train, y_test, withpca, i)
        print("fot svm, training time, testing time, accuracy")
        print(svm_triningTime, svm_testingTime, svm_accuracy)
        print("fot knn, training time, testing time, accuracy")

        print(knn_triningTime, knn_testingTime, knn_accuracy)
        print("fot LR, training time, testing time, accuracy")

        print(LR_triningTime, LR_testingTime, LR_accuracy)
        print("fot DT, training time, testing time, accuracy")

        print(DT_triningTime, DT_testingTime, DT_accuracy)
        svm_TriningTime, svm_TestingTime, svm_Accuracy = max(svm_TriningTime, svm_triningTime), max(svm_TestingTime, svm_testingTime), max( svm_Accuracy, svm_accuracy)
        knn_TriningTime, knn_TestingTime, knn_Accuracy = max(knn_TriningTime, knn_triningTime), max(knn_TestingTime,
                                                                                                    knn_testingTime), max(
            knn_Accuracy, knn_accuracy)
        LR_TriningTime, LR_TestingTime, LR_Accuracy = max(LR_TriningTime, LR_testingTime), max(LR_TestingTime,
                                                                                               LR_testingTime), max(
            LR_Accuracy, LR_accuracy)
        DT_TriningTime, DT_TestingTime, DT_Accuracy = max(DT_TriningTime, DT_triningTime), max(DT_TestingTime,
                                                                                               DT_testingTime), max(
            DT_Accuracy, DT_accuracy)

    ###########bar graphs
    model_name = ('SVM', 'KNN', 'Logistic Regression', 'Decision Tree')
    y_pos = np.arange(len(model_name))
    training_time = [svm_TriningTime, knn_TriningTime, LR_TriningTime, DT_TriningTime]
    testing_time = [svm_TestingTime, knn_TestingTime, LR_TestingTime, DT_TestingTime]
    accuracy = [svm_Accuracy, knn_Accuracy, LR_Accuracy, DT_Accuracy]

    plt.bar(y_pos, training_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Training Time')
    plt.show()

    plt.bar(y_pos, testing_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Testing Time')
    plt.show()

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('accuracy')
    plt.title('Acuuracy')
    plt.show()