from sklearn.feature_extraction.text import TfidfVectorizer
import pprint

frase = ["Hola mi nombre es Carles y tengo 24 años"]
tf = TfidfVectorizer()
vectorizado = tf.fit_transform(frase)

print(vectorizado)