import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def load_dataset(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))

    for file in feature_paths:
        file = "../../dataset/classifier/body_state_dataset/" + file
        df = pd.read_table(file, delimiter=",", na_values="?", header=None)
        imp = SimpleImputer( strategy="mean")
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))

    for file in label_paths:
        file = "../../dataset/classifier/body_state_dataset/" + file
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label

if __name__ == '__main__' :
    #设置数据路径
    feature_paths = ["A/A.feature","B/B.feature","C/C.feature","D/D.feature","E/E.feature"]
    label_paths = ["A/A.label","B/B.label","C/C.label","D/D.label","E/E.label"]

    #将前四个数据作为训练集读入
    x_train, y_train = load_dataset(feature_paths[:4], label_paths[:4])
    #将最后一个数据作为测试集读入
    x_test, y_test = load_dataset(feature_paths[4:], label_paths[4:])

    #使用全局数据作为训练集，借助train_test_split函数将训练数据打乱
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 1)

    #创建k近邻分类器，并在测试集上预测
    print("Start training knn")
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print("Training done!")
    answer_knn = knn.predict(x_test)
    print("Prediction done!")

    #创建决策树分类器，并在测试集上预测
    print("Start training DT")
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print("Training done!")
    answer_dt = dt.predict(x_test)
    print("Prediction done!")

    #创建贝叶斯分类器，并在测试集上预测
    print("Start training Bayes")
    gnb = GaussianNB().fit(x_train, y_train)
    print("Training done!")
    answer_gnb = gnb.predict(x_test)
    print("Prediction done!")

    #计算准确率与召回率
    print("\n\nThe classification report for knn:")
    print(classification_report(y_test, answer_knn))

    print("\n\nThe classification report for dt:")
    print(classification_report(y_test, answer_dt))

    print("\n\nThe classification report for gnb:")
    print(classification_report(y_test, answer_gnb))
