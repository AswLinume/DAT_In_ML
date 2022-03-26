import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath,'r+',encoding='gbk')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        temp =[]
        for i in range(1,len(items)):
            temp.append(float(items[i]))
        retData.append(temp)
    print(retData)
    print(retCityName)
    return retData,retCityName


if __name__ == '__main__':
    data,cityName = loadData('../../dataset/cluster/provice_comsumption_index.txt')
    km = KMeans(n_clusters=3)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    print(expenses)
    CityCluster = [[],[],[]]
    print(label)
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f"%expenses[i])
        print(CityCluster[i])

