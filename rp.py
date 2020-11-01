import time
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.util import scale_data, plot_clusters

np.random.seed(93)

def red_wine_quality(method='km'):
    if method == 'km':
        # Read in Red Wine Quality
        red_wine_data_km = pd.read_csv('data/red-wine-quality/full.csv')

        # Label encode/transform
        red_wine_data_km['quality'] = pd.cut(red_wine_data_km['quality'], bins=[2, 5.5, 8], labels=['bad', 'good'])
        le = LabelEncoder()
        red_wine_data_km['quality'] = le.fit_transform(red_wine_data_km['quality'])

        # Group x, y
        x = red_wine_data_km.drop('quality', axis=1).values
        y = red_wine_data_km['quality']

        create_start = time.process_time()

        # Randomized Projection
        c = 2
        for i in range(1000):
            x = GaussianRandomProjection(n_components=c).fit_transform(x)

        test_train_start = time.process_time()

        # Test-train split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        x_train, x_test = scale_data(x_train, x_test)

        test_train_time = time.process_time() - test_train_start

        # K-Means
        k = 2
        kmeans = KMeans(n_clusters=k).fit(x_train)
        
        run_time = time.process_time() - create_start - test_train_time
        print('RWQ [RP + k-means] time (ms):', run_time*1000)

        km_as = accuracy_score(kmeans.predict(x_test),y_test)
        print("k-means clustering accuracy score: ",km_as)

        # Plot
        plot_clusters(k, x_test, kmeans, 'dimen-reduction/rp/red-wine-k-means')
    elif method == 'em':
        # Read in Red Wine Quality
        red_wine_data_em = pd.read_csv('data/red-wine-quality/full.csv')

        # Label encode/transform
        red_wine_data_em['quality'] = pd.cut(red_wine_data_em['quality'], bins=[2, 5.5, 8], labels=['bad', 'good'])
        le = LabelEncoder()
        red_wine_data_em['quality'] = le.fit_transform(red_wine_data_em['quality'])

        # Group x, y
        x = red_wine_data_em.drop('quality', axis=1).values
        y = red_wine_data_em['quality']

        create_start = time.process_time()

        # Randomized Projection
        c = 3
        for i in range(1000):
            x = GaussianRandomProjection(n_components=c).fit_transform(x)

        test_train_start = time.process_time()

        # Test-train split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        x_train, x_test = scale_data(x_train, x_test)

        test_train_time = time.process_time() - test_train_start

        # EM
        k = 2
        em = GaussianMixture(n_components=k).fit(x_train)

        run_time = time.process_time() - create_start - test_train_time
        print('RWQ [RP + EM] time (ms):', run_time*1000)

        em_as = accuracy_score(em.predict(x_test),y_test)
        print("EM clustering accuracy score: ",em_as)

        # Plot
        plot_clusters(k, x_test, em, 'dimen-reduction/rp/red-wine-em')
    else:
        print('Invalid method: {}'.format(method))

def fish_market(method='km'):
    if method == 'km':
        # Read in Fish market
        fish_data = pd.read_csv('data/fish-market/full.csv')
        species = fish_data['Species'].value_counts().index.tolist()

        # Group x, y
        x, y = fish_data.drop('Species', axis=1).values, []
        for fish in fish_data['Species']:
            y.append(species.index(fish))

        create_start = time.process_time()

        # Randomized Projection
        c = 3
        for i in range(1000):
            x = GaussianRandomProjection(n_components=c).fit_transform(x)

        test_train_start = time.process_time()

        # Test-train split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
        x_train, x_test = scale_data(x_train, x_test)

        test_train_time = time.process_time() - test_train_start

        # K-Means
        k = 3
        kmeans = KMeans(n_clusters=k).fit(x_train)

        run_time = time.process_time() - create_start - test_train_time
        print('Fish market [RP + k-means] time (ms):', run_time*1000)

        km_as = accuracy_score(kmeans.predict(x_test),y_test)
        print("k-means clustering accuracy score: ",km_as)

        # Plot
        plot_clusters(k, x_test, kmeans, 'dimen-reduction/rp/fish-market-k-means')
    elif method == 'em':
        # Read in Fish market
        fish_data = pd.read_csv('data/fish-market/full.csv')
        species = fish_data['Species'].value_counts().index.tolist()

        # Group x, y
        x, y = fish_data.drop('Species', axis=1).values, []
        for fish in fish_data['Species']:
            y.append(species.index(fish))

        create_start = time.process_time()

        # Randomized Projection
        c = 6
        for i in range(1000):
            x = GaussianRandomProjection(n_components=c).fit_transform(x)

        test_train_start = time.process_time()

        # Test-train split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
        x_train, x_test = scale_data(x_train, x_test)

        test_train_time = time.process_time() - test_train_start

        # EM
        k = 3
        em = GaussianMixture(n_components=k).fit(x_train)

        run_time = time.process_time() - create_start - test_train_time
        print('Fish market [RP + EM] time (ms):', run_time*1000)

        em_as = accuracy_score(em.predict(x_test),y_test)
        print("EM clustering accuracy score: ",em_as)

        # Plot
        plot_clusters(k, x_test, em, 'dimen-reduction/rp/fish-market-em')
    else:
        print('Invalid method: {}'.format(method))

if __name__ == "__main__":
    if len(sys.argv) > 3:
        print('Too many arguments provided: {} ({})'.format(sys.argv, len(sys.argv)))
        print('Proper usage: rp.py <dataset> [red_wine_quality|fish_market] <method> [km|em]')
    elif len(sys.argv) < 3:
        print('Too few arguments provided: {} ({})'.format(sys.argv, len(sys.argv)))
        print('Proper usage: rp.py <dataset> [red_wine_quality|fish_market] <method> [km|em]')
    else:
        dataset = str(sys.argv[1])
        method = str(sys.argv[2])

        if dataset == 'red_wine_quality':
            red_wine_quality(method)
        elif dataset == 'fish_market':
            fish_market(method)
        else:
            print('Invalid dataset: {}'.format(dataset))