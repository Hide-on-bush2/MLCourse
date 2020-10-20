# from sklearn.datasets import load_iris
# from sklearn.datasets import make_classification
# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

# Label = {0:"A", 1:"B", 2:"C"}

# def createDataSet():
#     # iris = load_iris()
#     # return iris.data, iris.target
#     group, labels = make_classification(n_samples=100, n_informative=2, n_redundant=0, n_repeated=0, n_features=2, n_classes=3, n_clusters_per_class=1, scale=10)
#     # print(data)
#     return group, labels

# def classify(sample, group, labels, train_num):
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(group, labels)
#     result = knn.predict([sample])
#     return Label[result[0]]


# if __name__ == "__main__":
#     group, labels = createDataSet()
#     print("group:", group)
#     print("labels:", labels)
#     print("result:", classify([0, 0], group, labels, 3))


# -*- coding=utf-8 -*-


import numpy as np
import operator  

#创建简单数据集
def createDataSet():

    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
    
# k近邻算法
def classify(inX,dataSet,labels,k): 

    # inX用于分类的输入向量 
    # dataSet输入的训练样本集
    # labels为标签向量 
    # k为最近的邻居数目
    
    # 计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1) 
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1 
    
    # 排序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse= True) 
    return sortedClassCount[0][0]
