import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.util import scale_data

def fish_market():
    np.random.seed(93)

    fish_data1, fish_data2 = pd.read_csv('data/fish-market/full.csv'), pd.read_csv('data/fish-market/full.csv')
    x1, x2 = fish_data1.drop('Species', axis=1).values, fish_data2.drop(['Species', 'Weight'], axis=1).values
    y1 = fish_data1['Species']
    
    # Label encode/transform
    fish_data2['Weight'] = pd.cut(fish_data2['Weight'], bins=[0, 120, 650, 1650], labels=['light', 'avg', 'heavy'])
    le = LabelEncoder()
    fish_data2['Weight'] = le.fit_transform(fish_data2['Weight'].astype(str))

    y2 = fish_data2['Weight']

    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.25, random_state=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.25, random_state=1)
    x_train1, x_test1 = scale_data(x_train1, x_test1)
    x_train2, x_test2 = scale_data(x_train2, x_test2)

    k1 = KNeighborsClassifier(n_neighbors=5).fit(x_train1, y_train1)

    k1_as = accuracy_score(k1.predict(x_test1), y_test1)
    print("k1 accuracy score: ",k1_as)

    k2 = KNeighborsClassifier(n_neighbors=5).fit(x_train2, y_train2)

    k2_as = accuracy_score(k2.predict(x_test2), y_test2)
    print("k2 accuracy score: ",k2_as)

if __name__ == "__main__":
    fish_market()