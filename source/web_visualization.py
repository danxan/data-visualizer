# General
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, flash, redirect, url_for
import uuid

# Scikit-learn Models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Scikit-learn Tools
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# My own modules
from data import read, split_proportional
from visualize import visualize
from fitting import select_features, fit

app = Flask(__name__)

# Initial Route
@app.route("/")
def set_parameters():
    return render_template('select.html')

# Plotting and showing Plot
@app.route("/plot", methods=['POST'])
def main():
    try:
        dataset = request.form["dataset"]
        if(dataset == "static/diabetes.csv"):
            target = "diabetes"

        feature1 = request.form["feature1"]
        feature2 = request.form["feature2"]

        classifier = request.form["classifier"]
    except KeyError:
        error = "Warning! Missing selections. Please select one dataset, two features from the dataset, and one classifier!"
        return render_template('select.html', error=error)

    df = read(dataset)
    X_train, X_test, y_train, y_test = split_proportional(df, target)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    clf = fit(X_train, y_train, classifier, feature1, feature2)

    data_train = select_features(X_train, (feature1, feature2))
    data_test = select_features(X_test, (feature1, feature2))
    accuracy_train = np.mean(cross_val_score(clf, data_train, y_train, cv=5))
    accuracy_test = np.mean(cross_val_score(clf, data_test, y_test, cv=5))

    plot_data = build_plot(data_test, y_test, clf)

    return render_template('plot.html', accuracy_train=accuracy_train, accuracy_test=accuracy_test, plot_url=plot_data)

# Help page
def build_plot(data, y, clf):
    fig = visualize(data, y, clf)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.savefig(output, format='png')
    output.seek(0)

    plot_data = base64.b64encode(output.getvalue()).decode()

    return plot_data

@app.route("/help")
def help():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)

