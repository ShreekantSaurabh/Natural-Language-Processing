# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
#read.delim is used to read tsv file
#quote = '' to disable/ignore quotes
#stringsAsFactors = FALSE means reviews shouldn't be identified as factor/categorical. It must be numeric.

# Cleaning the texts
#install.packages('tm')         #tm package is used for NLP
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))  #VectorSource has parameter which we want to clean ie. the review column
corpus = tm_map(corpus, content_transformer(tolower))   #convert everything to lower case alphabet
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())      #remove stopwords like "the", "a", "an", "in" etc.
corpus = tm_map(corpus, stemDocument)                  #stem is used to find root of the word
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)     #Creates Sparse Matrix  
dtm = removeSparseTerms(dtm, 0.999)  #only 99.9% most frequently used words are chosen
dataset = as.data.frame(as.matrix(dtm))   #converting independent variables into dataframe
dataset$Liked = dataset_original$Liked    #dependent variable ie. reviews are positive or negative

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
#692 is column no. of dependent variable

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)