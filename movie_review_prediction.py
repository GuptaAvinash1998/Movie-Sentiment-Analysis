# Avinash Gupta
# G01009180
# CS 484 002
# Project 1
# This is a project that reads a files containing movie reviews and then accepts a new review and determines whether
# it is a good or bad review by using KNN
import string
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

correct_prediction = 0
nearest_labels = []


def knn_sentiment(train_reviews, test_reviews, oneslabels):

	cos_similarity = cosine_similarity(train_reviews,test_reviews).flatten()
	c = np.asarray(cos_similarity)
	i = 0

	while i < len(cos_similarity):

		nearest_labels.append(oneslabels[i])
		i += 1
		
	i = 0
	#nearest_labels.sort(reverse=True)
	#print(len(nearest_labels))
	
	return nearest_labels
	
review_file = open("train.txt", "r",encoding="utf-8") #opens the file
test_file = open("test.txt", "r",encoding="utf-8") #opens the test file
labels_file = open("labels.txt","w",encoding="utf-8")

reviews = review_file.readlines() #reads all the lines
test_reviews = test_file.readlines() #reads all the lines

stop_words = set(stopwords.words('english')) #gets the list of stop words defined in the library and stores it in the list named stop_words


tab = str.maketrans("", "", string.punctuation) #This makes a translation table. It takes 3 parametrs, the first is a string that specifies the characters that need to be replaced
												  #The second is a string that contains the characters that replace the characters that need to be replaced
												  #The thrid is a string that contains characters that need to be deleted
												  #In this case, we want to remove the punctuations

stemmer = LancasterStemmer()
doc_vector = TfidfVectorizer()

labels = []
i=0


while i<len(reviews):

	labels.append(reviews[i].split()[0]) #stores the label of that review
	
	reviews[i] = reviews[i].translate(tab)
	reviews[i] = reviews[i].replace("<br />", "")
	reviews[i] = reviews[i].replace("EOF", "")
	reviews[i] = reviews[i].replace("1", "")
	
	
	test_reviews[i] = test_reviews[i].translate(tab)
	test_reviews[i] = test_reviews[i].replace("<br />", "")
	test_reviews[i] = test_reviews[i].replace("EOF", "")
	
	copy = ""
	words = reviews[i].split()
	
	j=0
	
	while j<len(words):
		
		if words[j].lower() in stop_words:
			
			words.remove(words[j])
		else:
			copy += " " + words[j]#stemmer.stem(words[j])  
		j += 1
	#print("Training data: " + copy)
	reviews[i] = copy
	
	j=0
	
	copy = ""
	words = test_reviews[i].split()
	
	while j<len(words):
		
		if words[j].lower() in stop_words:
			
			words.remove(words[j])
		else:
			copy += " " + words[j] #stemmer.stem(words[j])  
		j += 1
	#print("Testing data: " + copy)	
	test_reviews[i] = copy
	i += 1

#vocab_matrix = np.concatenate((reviews,test_reviews), axis=0)
training = doc_vector.fit_transform(reviews)

i=0
accuracy_labels = []

while i<len(test_reviews):
	
	test_array = []
	test_array.append(test_reviews[i]);
	testing = doc_vector.transform(test_array)
	temp = knn_sentiment(training,testing,labels)
	accuracy_labels.append(max(temp))
	i+= 1

	
i=0
print(len(accuracy_labels))
while(i<25000):

	labels_file.write(accuracy_labels[i] + "\n")
	i+=1
	
review_file.close()
test_file.close()
labels_file.close()