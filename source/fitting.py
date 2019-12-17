import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from data import read, split_proportional, scatter_2cls
from visualize import visualize

def select_features(dataframe, features):
    '''
    Sets up a dataset for training, based on feature names given as arguments,
    if they are found in dataframe.columns.

    Args:
        * Pandas dataframe.
        * Iterable of strings. The desired traning features.

    Returns:
        Pandas dataframe of training data.
    '''

    if(len(features) == 0):
        print('No features chosen.')
        exit()
    else:
        features = list(features)

    try:
        # For f in features get the feature from dataframe
        df = dataframe[features]
    except KeyError:
        # If features was not found in df.columns then remove those from argument list
        remove = []
        for f in features:
            if not f in dataframe.columns:
                remove.append(f)

        print('The following arguments were not valid colmun names for the dataset:')
        print(remove)

        for r in remove:
            features.remove(r)

        if len(features) == 0:
            print('No valid features chosen.')
            exit()
        else:
            df = dataframe[features]

    return df


def fit(X, y, classifier, *features):
    '''
    A function to train a specified classifier on the specified features of a specified dataset.
    It is designed for users with an interface that outputs stdout and stderr.

    Args:
        X: Pandas dataframe containing training data
        y: Pandas dataframe containing target feature
        classifier: string. The program can output which strings are allowed.
        features: An iterable of strings to include in training.

    Returns:
        A trained classifier.
    '''
    # Building training data based on chosen features
    data = select_features(X, features)

    # Classifiers availale to the user
    clfs = {'mlp' : MLPClassifier, \
            'decisiontree' : DecisionTreeClassifier,\
            #'logreg' : LogisticRegression,\
            #'knn' : KNeighborsClassifier, \
            #'svm' : SVC, \
            'randomforest' :  RandomForestClassifier \
            #'adaboost' : AdaBoostClassifier, \
            #'naivebayes' : GaussianNB \
            }

    try:
        clf = clfs[classifier]() # Initialize all classifiers with default values
    except KeyError as e:
        print('Input to classifier parameter was '+ str(e))
        print('The only allowed argument(s) are:')
        for item in clfs.items():
            print('\t * '+item[0]+', for '+str(item[1]))
        exit()

    clf.fit(data,y)

    print("score on training data")
    print(cross_val_score(clf, data, y, cv=5))
    print("ok")

    return clf

if __name__ == '__main__':

    df = read('static/diabetes.csv')

    # Creating a test set to validate the accuracy of the different classifiers.
    X_train, X_test, y_train, y_test = split_proportional(df, "diabetes")

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    print("decisiontree")
    clf = fit(X_train, y_train, 'decisiontree',  'pedigree', 'glucose')
    data = select_features(X_test, ('pedigree', 'glucose'))
    test_pred = clf.predict(data)
    print("score on test data")
    print(accuracy_score(y_test, test_pred))
    print(cross_val_score(clf, data, y_test, cv=5))
    fig = visualize(data, y_test, clf)
    plt.show()



