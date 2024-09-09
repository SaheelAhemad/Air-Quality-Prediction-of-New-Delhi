import pandas as pd
import numpy as np
import matplotlib as mp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import plotly.express as px
import sklearn.metrics

# Define functions for data preprocessing
def fill_na(data):
    return data.fillna(0)

def map_categorical(data, column):
    unique_values = np.unique(data[column])
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    return data.replace({column: mapping})

# Define functions for visualizing data
def scatter_plot(x, y, xlabel, ylabel):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Define functions for model training and evaluation
def train_test_split(data, labels, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    test_indices = indices[:int(len(data) * test_size)]
    train_indices = indices[int(len(data) * test_size):]
    x_train, x_test = data[train_indices], data[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    return x_train, x_test, y_train, y_test

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define a simple random forest regressor
class RandomForestRegressor:
    def __init__(self, max_depth=2, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, x_train, y_train):
        np.random.seed(self.random_state)
        for _ in range(10):  # Let's train 10 trees for simplicity
            indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
            x_subset, y_subset = x_train[indices], y_train[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(x_subset, y_subset)
            self.trees.append(tree)

    def predict(self, x_test):
        predictions = np.zeros(len(x_test))
        for tree in self.trees:
            predictions += tree.predict(x_test)
        return predictions / len(self.trees)

# Define a simple decision tree regressor
class DecisionTreeRegressor:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def fit(self, x_train, y_train):
        self.tree = self._build_tree(x_train, y_train, depth=0)

    def _build_tree(self, x_train, y_train, depth):
        if depth == self.max_depth or len(np.unique(y_train)) == 1:
            return np.mean(y_train)
        else:
            feature_idx, threshold = self._find_best_split(x_train, y_train)
            left_indices = x_train[:, feature_idx] <= threshold
            right_indices = ~left_indices
            left_subtree = self._build_tree(x_train[left_indices], y_train[left_indices], depth + 1)
            right_subtree = self._build_tree(x_train[right_indices], y_train[right_indices], depth + 1)
            return {'feature_idx': feature_idx, 'threshold': threshold,
                    'left_subtree': left_subtree, 'right_subtree': right_subtree}

    def _find_best_split(self, x_train, y_train):
        best_mse = float('inf')
        best_feature_idx = None
        best_threshold = None
        for feature_idx in range(x_train.shape[1]):
            thresholds = np.unique(x_train[:, feature_idx])
            for threshold in thresholds:
                left_indices = x_train[:, feature_idx] <= threshold
                right_indices = ~left_indices
                if len(y_train[left_indices]) > 0 and len(y_train[right_indices]) > 0:
                    mse = self._calculate_mse(y_train[left_indices], y_train[right_indices])
                    if mse < best_mse:
                        best_mse = mse
                        best_feature_idx = feature_idx
                        best_threshold = threshold
        return best_feature_idx, best_threshold

    def _calculate_mse(self, left_labels, right_labels):
        total_samples = len(left_labels) + len(right_labels)
        mse = (len(left_labels) / total_samples) * np.var(left_labels) + \
              (len(right_labels) / total_samples) * np.var(right_labels)
        return mse

    def predict(self, x_test):
        predictions = []
        for sample in x_test:
            predictions.append(self._traverse_tree(sample, self.tree))
        return np.array(predictions)

    def _traverse_tree(self, sample, node):
        if isinstance(node, dict):
            if sample[node['feature_idx']] <= node['threshold']:
                return self._traverse_tree(sample, node['left_subtree'])
            else:
                return self._traverse_tree(sample, node['right_subtree'])
        else:
            return node

# Load data skipping the header row
pollution_data = np.genfromtxt('Delhi.csv', delimiter=',', dtype=str, skip_header=1)
header = pollution_data[0]
data = pollution_data[1:].astype(float)


# Preprocess data
data = fill_na(data)
data = map_categorical(data, column=header.index('City'))

# Visualize data
scatter_plot(data[:, header.index('City')], data[:, header.index('AQI')], 'City', 'AQI')

# Split data into features and labels
features = data[:, [header.index(col) for col in ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                                                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]]
labels = data[:, header.index('AQI')]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

# Train a Random Forest Regressor model
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
