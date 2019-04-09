import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.metrics import accuracy_score


def knn_sentiment(training_idf, testing_vector, train_labels, k=1000):
    nearest_labels = []
    cos_mat = cosine_similarity(testing_vector, training_idf)  # makes a cosine similarity matrix
    i = 0

    while i < cos_mat.shape[0]:
        sample = cos_mat[i]  # stores the pair of the ith testing review with all the training reviews
        sample = np.argsort(sample)  # sorts the list through the indices
        sample = sample[::-1]  # flips the list so it looks like descending order
        index = sample[:k]  # gets the k nearest labels
        most_occuring_label = Counter(index)
        nearest_labels.append(train_labels[most_occuring_label.most_common(1)[0][0]])
        i += 1
    return nearest_labels

train_file = open("training_reviews.txt", "r", encoding="utf-8")
predicted_file = open("labels.txt", "w")
train = train_file.readlines()  # reads all the lines in the training file

stopwords = set(stopwords.words('english'))  # gets the list of stop words

tab = str.maketrans("", "", string.punctuation)  # removes all the punctuation

doc_vector = TfidfVectorizer()
doc_stemmer = PorterStemmer()

labels = []
i = 0

while i < len(train):  # cleans the training reviews

    labels.append(train[i].split()[0])  # splits the review and stores the label of that review in the list
    train[i] = train[i].translate(tab)  # removes the punctuation
    train[i] = train[i].replace("<br \>", "")  # removes the break tag
    train[i] = train[i].replace("EOF", "")  # removes the EOF marker
    train[i] = train[i].replace("1", "")  # removes the label of the review

    words = train[i].split()
    copy = ""

    j = 0

    while j < len(words):
        if words[j].lower() in stopwords:
            words.remove(words[j])
        else:
            copy += words[j] + " "  # removes all the stop words and makes a new string
        j += 1

    train[i] = copy  # replaces the cleaned line into the review list
    i += 1

i = 0

divider = (2 * len(train)) // 3

training_half = train[:divider - 1]
training_labels = labels[:divider - 1]

testing_half = train[divider:]
testing_labels = labels[divider:]

training = doc_vector.fit_transform(training_half)  # converts the training reviews to a tf idf matrix
testing = doc_vector.transform(testing_half)  # converts the test reviews to a tf idf matrix

predicted_labels = knn_sentiment(training, testing, labels)

i = 0

accuracy = accuracy_score(testing_labels, predicted_labels)

while i < len(predicted_labels):
    predicted_file.write(predicted_labels[i] + "\n")
    i += 1

print("The number of accuracy of the model is " + str(accuracy*100))
print("project completed")
train_file.close()
predicted_file.close()
