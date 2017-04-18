import sys
import time
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

o = []

def load_data():
    data = [l.strip() for l in open('iris.data') if l.strip()]
    features = [tuple(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return dict(zip(features, labels))

def dist(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.sqrt(np.dot(d, d))
    # dist 為計算多維空間中兩資料點的實際距離
    # 將 tuple 轉為 np.array 並使用 np 的 sqrt, dot 函式

def dist2(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
    return np.dot(d, d)
    # dist2 為計算 variance (差平方)

def mean(clusters):
    return tuple(np.mean(clusters, axis=0))
    # mean 為計算某一群集的平均

def assign(centers):
    new_centers = defaultdict(list)
    for cx in centers:
        for x in centers[cx]:
            best = min(centers, key=lambda c: dist(x,c))
            new_centers[best] += [x]
    return new_centers
    # 呼叫 dist 函式來作為比較形心得依據

def update(centers):
    new_centers = {}
    for c in centers:
        print(c)
        print(mean(centers[c]))
        new_centers[mean(centers[c])] = centers[c]
    return new_centers
    # 更新形心(計算平均值)

def kmeans(features, k, maxiter=100):
    sse = []
    centers = dict((c,[c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    for i in range(maxiter):
        print(i)
        new_centers = assign(centers)
        new_centers = update(new_centers)
        countSSE(new_centers, sse)
        print(sse)
        if centers == new_centers:
            break
        else:
            centers = new_centers
        showPlot(centers)  
    plt.close()
    showSSE(sse)
    return centers

def counter(alist):
    count = defaultdict(int)
    for x in alist:
        count[x] += 1
    return dict(count)
    # 單純是個 cluster 的計數器，計算經分群的結果

def demo(seed=999):
    try:
        data = load_data()         # data 為 dict type
    except IOError:
        print("Missing dataset! Run:")
        print("wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
        sys.exit(1)
    features = list(data.keys())   # features 為 list of POINTS
    countTrueMean(features)
    random.seed(seed)
    random.shuffle(features)       # 打亂原本的順序
    clusters = kmeans(features, 3) # clusters 為dict{key(tuple)=形心座標: value(list)=該群的點} 
    for c in clusters:
        print(counter([data[x] for x in clusters[c]]))
    # showPlot(clusters)


# ======================  以下是 pyplot 繪圖  ========================
# ======================      code 髒髒的     ========================

def countTrueMean(features):
    global o
    o += [mean(features[0:50])]
    o += [mean(features[50:100])]
    o += [mean(features[100:150])]
    o = np.array(o)

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
    listCenters = np.array(listCenters)
    listPoints = np.array(listPoints)
    plt.scatter(listPoints[0:numbers_K[1], 0], listPoints[0:numbers_K[1], 2], color = 'red',label = 'Center 1')
    plt.scatter(listPoints[numbers_K[1]:numbers_K[1]+numbers_K[2], 0], listPoints[numbers_K[1]:numbers_K[1] + numbers_K[2], 2], color = 'blue', label = 'Center 2')
    plt.scatter(listPoints[numbers_K[1]+numbers_K[2]:numbers_K[1]+numbers_K[2]+numbers_K[3], 0], listPoints[numbers_K[1]+numbers_K[2]:numbers_K[1]+numbers_K[2]+numbers_K[3], 2], color = 'purple', label = 'Center 3')
    plt.scatter(listCenters[:, 0], listCenters[:, 2], color = 'black', marker = 'D', label = 'Center')
    plt.scatter(o[:, 0], o[:, 2], color = 'c', marker = 'P', label = 'TrueMean')
    # print(listCenters)
    plt.xlabel('Sepal length') # [Sepal length, Sepal width, Petal length, Petal width]
    plt.ylabel('Petal length')
    plt.legend(loc = 'upper left')
    plt.draw()
    plt.pause(0.1)
    plt.clf()

def countSSE(centers, sse):
    dist2cxSum = 0
    for cx in centers:
        for x in centers[cx]:
            dist2cxSum += dist2(x, cx)
    sse += [dist2cxSum]
    return sse
def showSSE(sse):
    plt.xlabel('Iterations')
    plt.ylabel('WCSS')
    plt.title('Within-cluster Sum of Squares (Variance)')
    plt.plot(sse)
    plt.axis([0, len(sse), 50, 150])
    plt.show()
    plt.clf()
    return 0

if __name__ == "__main__":
    demo(18) 
