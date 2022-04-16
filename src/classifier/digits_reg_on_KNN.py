import numpy as np
from os import listdir
from sklearn import neighbors

def img2vector(filename):
    retMat = np.zeros([1024], int)
    fr = open(filename)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i * 32 + j] = lines[i][j]
    return retMat


def readDataSet(path):
    fileList = listdir(path)
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles])
    for i in range(numFiles):
        filePath = fileList[i]
        digit = int(filePath.split('_')[0])
        hwLabels[i] = digit
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels

train_dataSet, train_hwLabels = readDataSet('../../dataset/digits/trainingDigits')

knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)

dataSet, hwLabels = readDataSet('../../dataset/digits/testDigits')

res = knn.predict(dataSet)
error_num = np.sum(res != hwLabels)
num = len(dataSet)

print("Total num:", num, "Wrong num:", \
        error_num, "Wrong rata:", error_num / float(num))

