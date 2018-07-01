from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# Note: technically multinomial NB is for "counts", but the documentation says
#       it will work for other types of "counts", like tf-idf, so it should
#       also work for our "word proportions"

data = pd.read_csv('spambase.data').as_matrix() # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

X = data[:,:48]    #1st 48 columns are input
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]     #ie. X[0:4501, :]
Ytrain = Y[:-100,]     #ie. Y[0:4501, :]
Xtest = X[-100:,]      #ie. X[4501:4600, :]
Ytest = Y[-100:,]      #ie. Y[4501:4600, :]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for Naive Bayes:", model.score(Xtest, Ytest))

##### you can use ANY model! #####
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))