from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def createDataSet():
    # iris = load_iris()
    # return iris.data, iris.target
    group, labels = make_classification(n_samples=100, n_informative=2, n_redundant=0, n_repeated=0, n_features=2, n_classes=3, n_clusters_per_class=1, scale=10)
    # print(data)
    return group, labels

def classify(sample, group, labels, train_num):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(group, labels)
    return knn.predict([sample]) 

if __name__ == "__main__":
    group, labels = createDataSet()
    print("group:", group)
    print("labels:", labels)
    print("result:", classify([0, 0], group, labels, 3))