from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    knn = KNeighborsClassifier(n_neighbors=1)
    X = [[1, 2, 3],[11, 12, 13]]
    y = [0, 1]
    knn.fit(X, y)
    