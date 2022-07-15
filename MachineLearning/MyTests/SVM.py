import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel(r'C:\Users\Carles\PycharmProjects\MachineLearning\Data.xlsx')
x_red = np.array(df[df["Cura(S/N)"] == 1]["Dosis(mg)"])
x_blue = np.array(df[df["Cura(S/N)"] == 0]["Dosis(mg)"])

y_red = np.array(df[df["Cura(S/N)"] == 1]["Cura(S/N)"])
y_blue = np.array(df[df["Cura(S/N)"] == 0]["Cura(S/N)"])

# plt.scatter(x_red, y_red, marker='v', s=50, c='#fcc603', edgecolors='black')
# plt.scatter(x_blue, y_blue, s=50, marker='o', c='blue', edgecolors='black')
plt.axis([-0.5, 4, -2, 2])
#

X = np.array(df["Dosis(mg)"]).reshape(-1, 1)
y = np.array(df["Cura(S/N)"])

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=100, shuffle=True)

classifier = svm.SVC(C=1.5)
classifier.fit(xtrain, ytrain)

# plot_decision_regions(X, y, clf=classifier)
ypred = classifier.predict(xtest)
plt.scatter(xtest, ypred, marker='o', c='green',edgecolors='black')
plt.scatter(xtest, ytest, marker='o', c='yellow',edgecolors='black')
plt.show()