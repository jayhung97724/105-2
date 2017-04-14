import sys
import time
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_data():
    data = [l.strip() for l in open('iris.data') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return dict(zip(features, labels))

def dist2(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(np.dot(d, d))

def mean(feats):
    return tuple(np.mean(feats, axis=0))

def assign(centers):
    new_centers = defaultdict(list)
    for cx in centers:
        for x in centers[cx]:
            best = min(centers, key=lambda c: dist2(x,c))
            new_centers[best] += [x]
    return new_centers

def update(centers):
    new_centers = {}
    for c in centers:
        new_centers[mean(centers[c])] = centers[c]
    return new_centers

def kmeans(features, k, maxiter=100):
    centers = dict((c,[c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    for i in range(maxiter):
        print(i)
        new_centers = assign(centers)
        new_centers = update(new_centers)
        if centers == new_centers:
            break
        else:
            centers = new_centers
            showPlot(centers)
    return centers
def counter(alist):
    count = defaultdict(int)
    for x in alist:
        count[x] += 1
    return dict(count)

def demo(seed=999):
    try:
        data = load_data()         # data 為 dict type
    except IOError:
        print("Missing dataset! Run:")
        print("wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
        sys.exit(1)
    features = list(data.keys())   # features 為 list of POINTS
    random.seed(seed)
    random.shuffle(features)       # 打亂原本的順序
    clusters = kmeans(features, 3) # clusters 為dict{key(tuple)=形心座標: value(list)=該群的點} 
    for c in clusters:
        print(counter([data[x] for x in clusters[c]]))
    showPlot(clusters)

def showPlot(clusters):
    numbers_K = [0]
    listPoints = []
    listCenters = []
    centers = map(list, clusters.keys()) 
    points = map(list, clusters.values())
    for c in centers:
        listCenters += [c]
    for p in points:
        numbers_K += [len(p)]
        for t in p:
            listPoints += [[t[0], t[1], t[2], t[3]]]
    print(numbers_K[-1])
    listCenters = np.array(listCenters)
    listPoints = np.array(listPoints)
    plt.scatter(listPoints[0:numbers_K[1], 0], listPoints[0:numbers_K[1], 2], color = 'red',label = 'Center 1')
    plt.scatter(listPoints[numbers_K[1]:numbers_K[1]+numbers_K[2], 0], listPoints[numbers_K[1]:numbers_K[1] + numbers_K[2], 2], color = 'blue', label = 'Center 2')
    plt.scatter(listPoints[numbers_K[1]+numbers_K[2]:numbers_K[1]+numbers_K[2]+numbers_K[3], 0], listPoints[numbers_K[1]+numbers_K[2]:numbers_K[1]+numbers_K[2]+numbers_K[3], 2], color = 'purple', label = 'Center 3')
    plt.scatter(listCenters[:, 0], listCenters[:, 2], color = 'black', marker = 'D', label = 'Center')
    print(listCenters)
    plt.xlabel('Sepal length') # [Sepal length, Sepal width, Petal length, Petal width]
    plt.ylabel('Petal length')
    plt.legend(loc = 'upper left')
    plt.draw()
    plt.pause(3)
    plt.clf()

if __name__ == "__main__":
    demo(time) 
