def demo(seed=123):
    """
    The Iris dataset used in the demo is known to have a linearly separable
    class 'setosa' and 2 non linearly separable one each other.

    >>> demo()
    {'Iris-virginica': 1, 'Iris-versicolor': 29}
    {'Iris-virginica': 23}
    {'Iris-virginica': 25, 'Iris-versicolor': 21}
    {'Iris-setosa': 48}
    """
    data = load_data()
    features = data.keys()
    random.seed(seed)
    random.shuffle(features)
    clusters = kmeans(features, 4)
    for c in clusters:
        print counter([data[x] for x in clusters[c]])