from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer #Veces que aparece la palabra en doc
from sklearn.feature_extraction.text import TfidfTransformer #Tiene en cuenta la frecuencia inversa de aparicion en docs
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from pprint import pprint

categories = ['alt.atheism', 'sci.med'] #Solo textos de estos campos
training_data = fetch_20newsgroups(subset='train',
                                     categories=categories,
                                     shuffle=True,
                                     random_state=49,
                                     remove=('headers', 'footers', 'quotes'))

test_data = fetch_20newsgroups(subset='test',
                               categories=categories,
                               shuffle=True,
                               random_state=49,
                               remove=('headers', 'footers', 'quotes'))


# print("\n".join(training_data.data[0].split("\n")[:20]))
# print("Target is:", training_data.target_names[training_data.target[0]])

cv = CountVectorizer()
x_count_vector = cv.fit_transform(training_data.data)
# print(x_count_vector.shape)
# pprint(cv.vocabulary_)

tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(x_count_vector)

# print(tfidf_matrix)

model = MultinomialNB().fit(tfidf_matrix, training_data.target)

new = ["The removal of the patient's thorax was more difficult than expected", "The tumour had to be analyzed",
       "I'm not going to church anymore, I don't believe"]

x_new_count = cv.transform(test_data.data)
tfidf_new = tfidf_transformer.transform(x_new_count)

predicted = model.predict(tfidf_new)
head = ["\n".join(head.split("\n")[:15]) for head in test_data.data]
for doc, category in zip(head, predicted):
    print('%r -----> %s' % (doc, training_data.target_names[category]))

print("Accuracy score:", accuracy_score(test_data.target, predicted))
print("Counfussion Matrix:\n", confusion_matrix(test_data.target, predicted))


