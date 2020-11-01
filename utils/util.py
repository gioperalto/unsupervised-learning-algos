import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    return scaler.transform(x_train), scaler.transform(x_test)

def plot_clusters(k, x_test, algo, filename):
    predictions = algo.predict(x_test)
    a, b = 0, 0

    for i in range(k):
        plt.scatter(
            x_test[predictions == i, b],
            x_test[predictions == a, b+1],
            label='Cluster {}'.format(i)
        )
        a += 1
        if i % 2 == 1:
            b += 1

    plt.legend()
    plt.savefig('images/{}'.format(filename))
    plt.close()