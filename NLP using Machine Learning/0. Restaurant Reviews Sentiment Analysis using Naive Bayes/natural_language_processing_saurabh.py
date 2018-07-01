# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#delimiter = '\t' means two columns are seperated by a Tab ie. its a tsv file
#quoting = 3 is the code to ignore double quotes

# Cleaning the texts
import re        #Regular Expression
import nltk      #Natual language Tool Kit
#nltk.download()     #To download all packages of nltk
nltk.download('stopwords')      #use to remove stopwords like “the”, “a”, “an”, “in” etc.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer    #stem is used to find root of the word 
corpus = []        #Corpus is collection of texts of any type
for i in range(0, 1000):      #no. of reviews is 1000
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])     #Returns only a-z and A-Z, other non-alphabet letters will be removed and replaced by a white space. 
    review = review.lower()        #convert everything to lower case alphabet
    review = review.split()        #Split the review string into individual words ie. converted to a list.
    ps = PorterStemmer()           #object of PorterStemmer class
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model through Tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   #only 1500 most frequently used words are chosen
X = cv.fit_transform(corpus).toarray()     #Creating Sparse Matrix
Y = dataset.iloc[:, 1].values             #dependent variable ie. reviews are positive or negative

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)