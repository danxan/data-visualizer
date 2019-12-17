import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read(filename):
    '''
    Reads a csv and drops all ros with NaN values.

    Args: string
    Returns: pandas dataframe
    '''
    df = pd.read_csv(filename)
    df = df.dropna()

    return df

def split_proportional(df, target=None, test_size=0.2, train_size=None):
    '''
    Splits the dataset in a given train_test_split, and stratifies after the target column name.
    If target is provided, training and test will be separated into x and y.
    Returns the training and test dataframes, respectively.

    Args:
        dataframe: to be separated,
        string: target feature name,
        float: test size
        float: train size
    Returns: X_train, X_test, y_train, y_test
    '''
    if target != None:
        target = df[target]
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, train_size=train_size, stratify=target)
        return X_train, X_test, y_train, y_test


    train, test = train_test_split(df, test_size=test_size, train_size=train_size, stratify=target)
    return train, test

def scatter_2cls(df, cl1, cl2):
    '''
    Creates a scatter plot of two dimensions of the feature space,
    coloring the positive and negative outcomes differently.
    Diabetes positive is red, negative is blue.
    Positional arguments 1 and 2 is the names or position of the columns.

    Args: dataframe, string/int, string/int
    '''
    po = df['diabetes']=='pos'
    po = df[po]
    ne = df['diabetes']=='neg'
    ne = df[ne]
    ax1 = po.plot(x=cl1, y=cl2, color='r', kind='scatter')
    ax2 = ne.plot(x=cl1, y=cl2, color='b', kind='scatter', ax = ax1)


if __name__ == '__main__':
    df = read('diabetes')
    #print(df)
    x,y = split_proportional(df, 'diabetes')

    scatter_2cls(df, 'BloodPressure', 'glucose')


    #print(x.diabetes.value_counts())
   # print(y.diabetes.value_counts())










