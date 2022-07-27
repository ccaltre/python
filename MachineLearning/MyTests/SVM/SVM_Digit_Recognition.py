from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# print(digits.data.shape)
#print(digits.target)
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

images_and_labels = list(zip(digits.images, digits.target)) #Prepare data

# print(images_and_labels[0]) #Pixel intensity, digit

for index, (image, label) in enumerate(images_and_labels[:6]):
    plt.subplot(2, 3, index + 1) #Subplot with 2 rows and 3 columns
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Target: %i" % label)

plt.show()

#Hay que representar la data como una matriz de 64 columnas (una por pixel)
data = digits.images.reshape((len(digits.images), -1))
print(data.shape)
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, train_size=0.7, random_state=49)

model = svm.SVC(gamma=0.001)
model.fit(X_train, y_train)

print("Accuracy score: ", accuracy_score(y_test, model.predict(X_test)))
print(f"Confussion matrix: \n{metrics.confusion_matrix(y_test,model.predict(X_test))}")