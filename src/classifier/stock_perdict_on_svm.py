import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import model_selection

data = pd.read_csv('../../dataset/classifier/stock/000777.csv', encoding='gbk', parse_dates=[0], index_col=0)
data.sort_index(0, ascending=True, inplace=True)

dayfeature = 150
featurenum = 5 * dayfeature
x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
y = np.zeros((data.shape[0] - dayfeature))

for i in range(0, data.shape[0] - dayfeature):
    x[i, 0:featurenum] = np.array(data[i:i + dayfeature]\
            [[u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量']])\
            .reshape(1, featurenum)
            #extract closing price, top price, bottom price, opening price, turnover
    x[i,featurenum] = data.iloc[i + dayfeature][u'开盘价']
    #store opening price as last feature

for i in range(0, data.shape[0] - dayfeature):
    if data.iloc[i +  dayfeature][u'收盘价'] > data.iloc[i + dayfeature]['开盘价']:
        y[i] = 1
    else:
        y[i] = 0

clf = svm.SVC(kernel='rbf')
#other kernel function like linear、poly、sigmoid
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = model_selection\
            .train_test_split(x, y, test_size=0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))

print('svm classifier accuacy:')

print(result)
    
