def kmeans(features, k, maxiter=100):
    centers = dict((c,[c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    for i in xrange(maxiter):
        new_centers = assign(centers)
        new_centers = update(new_centers)
        if centers == new_centers:
            break
        else:
            centers = new_centers
    return centers