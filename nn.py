import time
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from utils.util import scale_data
from utils.plotter import Plotter

np.random.seed(93)

def create_nn(x, y, max_iter):
    return MLPClassifier(
        solver='sgd',
        random_state=0,
        hidden_layer_sizes=(x.shape[1]**2, x.shape[1], x.shape[1]**2),
        max_iter=max_iter,
        n_iter_no_change=1
    ).fit(x, y.values.flatten())

def create_nns(x, y, iter_counts):
    nns = []

    for count in tqdm(iter_counts):
        nn = create_nn(x=x, y=y, max_iter=count)
        nns.append(nn)

    return nns

def plot_nn(x, y, suffix=''):
    # Test-train split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_test = scale_data(x_train, x_test)

    iter_counts = np.arange(1, 1100, 50, dtype=int)

    # Create Red Wine Quality NNs
    nns = create_nns(x_train, y_train, iter_counts)

    # Get training/testing accuracy for Red Wine Quality
    accs_train, accs_test = [], []

    for nn in nns:
        accs_train.append(accuracy_score(nn.predict(x_train), y_train))
        accs_test.append(accuracy_score(nn.predict(x_test), y_test))
    
    # Generate graph for Red Wine Quality
    plot = Plotter(
        name='red-wine{}'.format(suffix), 
        learner='nn',
        axes={ 'x': 'Number of weight updates', 'y': 'Accuracy score' }
    )
    plot.add_plot(iter_counts, accs_train, 'training data', 'None')
    plot.add_plot(iter_counts, accs_test, 'testing data', 'None')
    plot.find_max(iter_counts, accs_test, 'testing')
    plot.save()

def transform_data():
    # Read in Red Wine Quality
    red_wine_data_km = pd.read_csv('data/red-wine-quality/full.csv')

    # Label encode/transform
    red_wine_data_km['quality'] = pd.cut(red_wine_data_km['quality'], bins=[2, 5.5, 8], labels=['bad', 'good'])
    le = LabelEncoder()
    red_wine_data_km['quality'] = le.fit_transform(red_wine_data_km['quality'])

    # Group x, y
    x = red_wine_data_km.drop('quality', axis=1).values
    y = red_wine_data_km['quality']

    return x, y

def pca(x):
    # PCA
    c = 4

    return PCA(n_components=c).fit_transform(x)

def k_means(x):
    # K-Means
    k = 2

    return KMeans(n_clusters=k).fit_transform(x)


if __name__ == "__main__":
    X, y = transform_data()

    plot_nn(X, y) # NN w/o PCA or k-means

    X = pca(X) # PCA dimensionality reduction applied to X
    plot_nn(X, y, '-pca') # NN w/ PCA applied

    X = k_means(X) # k-means clustering applied to X
    plot_nn(X, y, '-pca-k-means') # NN w/ PCA + k-means