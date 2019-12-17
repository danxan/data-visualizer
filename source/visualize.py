import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pylab as pl

from data import scatter_2cls

def visualize(X,y,clf):
    '''
    Plots the datapoints of the X dataframe, and the predictions of the provided classifier.
    Assuming the dataframe can only contain two features, while the number of predicted classes are unknown.

    Args:
        X: Pandas dataframe with two features for training.
        y: Pandas dataframe with the target classes for prediction.
        clf: A trained sklearn classifier.
    '''
    # Separating the possible classes of prediction
    y = pd.get_dummies(y)

    # Generating a list of colors that has the same size as the number of predictions
    cl=iter(plt.cm.rainbow(np.linspace(0,1,len(y.columns))))

    # Grabbing the a handle on the axes and figure of the plot
    fig, ax = plt.subplots()

    # Generating a contourplot
    f1 = X.iloc[:,0]
    f2 = X.iloc[:,1]
    f1_min, f1_max, f2_min, f2_max = f1.min()-.1, f1.max()+.1, f2.min()-.1, f2.max()+.1
    x_ = np.linspace(f1_min, f1_max, 1000)
    y_ = np.linspace(f2_min, f2_max, 1000)
    xx, yy = np.meshgrid(x_, y_)
    m = np.c_[xx.ravel(), yy.ravel()]
    z = clf.predict(m)
    z2 = z[:,1].reshape(xx.shape)
    mesh = ax.pcolormesh(xx,yy,z2)

    # Plots training data, with colors corresponding to the target class the data belongs to.
    for c in y.columns:
        data = X[y[c] == 1]
        ax.scatter(data.iloc[:,0], data.iloc[:,1], color=next(cl), label=c)

    # Set a colorbar with ticks corresponding to the classes of prediction
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_ticks(np.linspace(0,1,len(y.columns)))
    cb.set_ticklabels(y.columns)

    ax.legend()
    ax.grid(True)
    plt.subplots_adjust(top=0.80)
    fig.suptitle(str(clf), fontsize=7)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    return fig

