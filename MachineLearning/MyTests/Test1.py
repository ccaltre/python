import numpy as np
from sklearn.linear_model import LogisticRegression

lx1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

lx2 = [0.5, 0.8, 0.6, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0,
       0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

ly = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

x1 = np.array(lx1)
x2 = np.array(lx2)
y1 = np.array(ly)

X = np.array([lx1, lx2])
y = np.array(y1)

X = X.transpose()
# X = np.array(lx1+lx2).reshape(-1, 1)
# y = np.array(ly1+ly2)
#
# plt.plot(x1, y1, 'bo')
# plt.plot(x2, y2, 'ro')
# plt.axis([-2, 10, -0.5, 2])
# plt.show()

model = LogisticRegression()
model.fit(X, y)

print("b0 is: ", model.intercept_)
print("b1 is: ", model.coef_)

print("Predictions: ", model.predict(np.array([[0.5, 0.6, 0.3, 0.9], [0.7, 0.1, 0.9, 0.8]]).transpose()))

