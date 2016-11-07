from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from sklearn import svm
#from sklearn.model_selection import cross_val_score
f=open('Discussion_Category_Less2.csv')
csv_f = csv.reader(f)
clf = svm.SVC()

commentList = []
labelList = []
flag = 1
for row in csv_f:
	if flag == 1:
		flag = 0
		continue
	commentList.append(row[2])
	labelList.append(row[1])

vectorizer = CountVectorizer(min_df=1)
#print vectorizer
X = vectorizer.fit_transform(commentList).toarray()

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X)

# clf = svm.SVC(kernel='rbf')
# scores = cross_val_score(clf, X, labelList, cv=5)
# print scores


	
