# ASSIGNMENT 6
I have used sklearn, pandas and pyplot to solve most of the tasks in this assignment.

# 6.1: Handling the data
Most of this script is written specifically for the provided dataset "diabetes.csv".

To run: ```python data.py```

# 6.2: Fitting a machine learning model
This script was written to be generic for any dataset,
and because of this, the user is asked to provide the target column,
in addition to the features to be trained on.

I have tried to create output that is useful to a user interacting with the program through a terminal.

The current support for classifiers is limited to a multi-layer perceptron, decision-tree classifier and random forest.

When testing the accuracy of the classifiers for a variety of features,
the resulting scores seem rather random.

To run: ```python fitting.py```

# 6.3 Visualizing the classifier
The script works as expected, with a scatterplot over a contourplot.
This is tested in the main script of ```fitting.py```.

The code is uneccesarily complicated, as I chose to let predictions/targets be a dataframe with one column per class (pos/neg).
I have come to realize that this was a poor decision, but chose not to change it before I was happy with everything else,
and thus I never changed it.

The reason I don't use more classifiers is because most of them do not support multidimensional targets(, as far as I've understood).

To run: ```python fitting.py```

# 6.4 and 6.5 Interactive Visualization through a web app.
The app works according to the requirements of the task.
A lot of the features implemented in html could be done better with Flask/JavaScript,
but I never got the time to do this.

To run: ```python web_visualization.py```, and go to localhost:<given port>.

# 6.6 Documentation
Didn't finish this. I've generated a useless docs, and try to link to these from the plot page, but the linking is not done properly,
so it's all useless. At least I tried!
