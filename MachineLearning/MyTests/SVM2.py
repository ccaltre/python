import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_excel(r'C:\Users\Carles\PycharmProjects\MachineLearning\SVC_Classification.xlsx')

x1_orange = np.array(df[df["Y"] == 1]["X1"])
x2_orange = np.array(df[df["Y"] == 1]["X2"])

x1_blue = np.array(df[df["Y"] == 0]["X1"])
x2_blue = np.array(df[df["Y"] == 0]["X2"])

x1_purple = np.array(df[df["Y"] == 2]["X1"])
x2_purple = np.array(df[df["Y"] == 2]["X2"])

y_orange = np.array(df[df["Y"] == 1]["Y"])
y_blue = np.array(df[df["Y"] == 0]["Y"])

X = np.array(list(zip(df["X1"], df["X2"]))).reshape(-1, 2)
print("X Has shape :", X.shape)
y = np.array([df["Y"]]).reshape(-1,)
print("Y has shape: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=49)

plt.scatter(x1_orange, x2_orange, c='#ebc634', marker='o', edgecolors="black")
plt.scatter(x1_blue, x2_blue, c='#1752bf', marker='o', edgecolors='black')
plt.scatter(x1_purple, x2_purple, c='#7732a8', marker='o', edgecolors='black')
plt.axis([-4.5, 4.5, -4.5, 4.5])
plt.show()

classifier = svm.SVC(kernel='poly', degree=2)
classifier.fit(X_train, y_train)

print(classifier.predict([[1.3, 1.3]]))

plot_decision_regions(X, y, clf=classifier, legend=2)
plt.show()

print("Confussion matrix: \n", confusion_matrix(y_test, classifier.predict(X_test)))
print("Accuracy score: ", accuracy_score(y_test, classifier.predict(X_test)))
