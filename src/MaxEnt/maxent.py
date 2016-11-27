import csv
import nltk
import numpy
from textblob import TextBlob
from textblob.classifiers import MaxEntClassifier
from random import shuffle
# nltk.download()  ## for the first time
f = open("C:\\Users\\Simerdeep\\Desktop\\ncsu\\CS 522\\project\\AldaProject\\src\\CSV\\Discussion_Category_Less2.csv")
csv_f = csv.reader(f)
data = []
for r in csv_f:
	data.append([r[2].decode('utf-8'),r[1].decode('utf-8')])
data = data[1:]

## Split training,validation and test data in 60 20 20
shuffle(data)
train = data[:280]
test = data[280:]
cl = MaxEntClassifier(train)
print("Accuracy: {0}".format(cl.accuracy(test)))
