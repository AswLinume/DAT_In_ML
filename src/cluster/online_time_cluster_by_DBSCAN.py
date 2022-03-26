import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()
onlineTimes = []

f = open("../../dataset/cluster/online_data.txt")
lines = f.readlines();
for line in lines:
    items = line.split(',')
    mac = items[2]
    onlineTime = int(items[6])
    startTime = int(items[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac] = len(onlineTimes)
        onlineTimes.append((startTime, onlineTime))
    else:
        onlineTimes[mac2id[mac]] = [(startTime, onlineTime)]
#print("onlineTimes before reshape:%s"%onlineTimes)
real_X = np.array(onlineTimes).reshape((-1,2))
#print("onlineTimes after reshape:%s"%real_X)

X = real_X[:,0:1]
#print("onlineTimes after slice:%s"%X)

model = DBSCAN(eps = 0.01, min_samples = 20).fit(X)
labels = model.labels_

print("Labels:")
print(labels)
ratios = len(labels[labels[:] == -1]) / len(labels)
print("Noise ratios:", format(ratios, ".2%"))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print("Estimated number of clusters: %d"%n_clusters_)
print("Silhouette Coefficient: %0.3f"%metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print("Cluster ", i, ":")
    print(list(X[labels == i].flatten()))

#plt.hist(X, 24)
#plt.show()

X = np.log(1 + real_X[:,1:])
#print("onlineTimes after slice:%s"%X)

model = DBSCAN(eps = 0.14, min_samples = 10).fit(X)
labels = model.labels_

print("Labels:")
print(labels)
ratios = len(labels[labels[:] == -1]) / len(labels)
print("Noise ratios:", format(ratios, ".2%"))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print("Estimated number of clusters: %d"%n_clusters_)
print("Silhouette Coefficient: %0.3f"%metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print("Cluster ", i, ":")
    count = len(X[labels == i])
    mean = np.mean(real_X[labels == i][:,1])
    std = np.std(real_X[labels == i][:,1])
    print("\t number of sample: ", count)
    print("\t mean of sample  : ", format(mean, ".1f"))
    print("\t std of sampl   e: ", format(mean, ".1f"))

plt.xlabel('Online Time')
plt.hist(X, 20)
plt.show()
