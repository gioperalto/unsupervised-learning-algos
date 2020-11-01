import time
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.util import scale_data, plot_clusters

np.random.seed(93)

def red_wine_quality():
    # Read in Red Wine Quality
    red_wine_data = pd.read_csv('data/red-wine-quality/full.csv')

    # Label encode/transform
    red_wine_data['quality'] = pd.cut(red_wine_data['quality'], bins=[2, 5.5, 8], labels=['bad', 'good'])
    le = LabelEncoder()
    red_wine_data['quality'] = le.fit_transform(red_wine_data['quality'])

    # Test-train split
    x = red_wine_data.drop('quality', axis=1).values
    y = red_wine_data['quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_test = scale_data(x_train, x_test)

    create_start = time.process_time()

    # EM
    k = 2
    em = GaussianMixture(n_components=k).fit(x_train)

    run_time = time.process_time() - create_start
    print('RWQ [EM] time (ms):', run_time*1000)

    em_as = accuracy_score(em.predict(x_test),y_test)
    print("EM clustering accuracy score: ",em_as)

    # Plot
    plot_clusters(k, x_test, em, 'clustering/em/red-wine')

def fish_market():
    # Read in Fish market
    fish_data = pd.read_csv('data/fish-market/full.csv')
    species = fish_data['Species'].value_counts().index.tolist()

    # Test-train split
    x, y = fish_data.drop('Species', axis=1).values, []
    for fish in fish_data['Species']:
        y.append(species.index(fish))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    x_train, x_test = scale_data(x_train, x_test)

    create_start = time.process_time()

    # EM
    k = 3
    em = GaussianMixture(n_components=k).fit(x_train)

    run_time = time.process_time() - create_start
    print('Fish market [EM] time (ms):', run_time*1000)

    em_as = accuracy_score(em.predict(x_test),y_test)
    print("EM clustering accuracy score: ",em_as)

    # Plot
    plot_clusters(k, x_test, em, 'clustering/em/fish-market')

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print('Too many arguments provided: {} ({})'.format(sys.argv, len(sys.argv)))
        print('Proper usage: em.py <dataset> [red_wine_quality|fish_market]')
    elif len(sys.argv) < 2:
        print('Too few arguments provided: {} ({})'.format(sys.argv, len(sys.argv)))
        print('Proper usage: em.py <dataset> [red_wine_quality|fish_market]')
    else:
        dataset = str(sys.argv[1])

        if dataset == 'red_wine_quality':
            red_wine_quality()
        elif dataset == 'fish_market':
            fish_market()
        else:
            print('Invalid dataset: {}'.format(dataset))