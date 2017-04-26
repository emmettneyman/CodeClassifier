
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from os import listdir


data_list = []
target_list = []
for f in listdir("training_data"):
    with open("training_data/" + f, "r") as fp:
        target = [fp.readline().strip()]
        for line in fp:
            line = line.strip(' \t\n')
            if not line or line[0] == "#"\
                    or line[0] == "/"\
                    or line[0] == "*"\
                    or line[0] == "-"\
                    or line[0] == "{"\
                    or line[0] == "}":
                continue
            else:
                data_list.append(line)
                target_list.append(target)

data_array = np.array(data_list)


class classifier():

    history = []

    def __init__(self, targetlist, dataarray):
        self.mlb = MultiLabelBinarizer()
        self.Y = self.mlb.fit_transform(targetlist)
        self.classifier = Pipeline([
                        ('vectorizer', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', OneVsRestClassifier(LinearSVC()))])
        self.classifier.fit(dataarray, self.Y)

    def predictCode(self, xtest):
        X_test = np.array([xtest])
        self.predicted = self.classifier.predict(X_test)
        self.all_labels = self.mlb.inverse_transform(self.predicted)
        self.history.append((xtest, self.all_labels))

    def returnPrediction(self):
        for item in self.all_labels:
            if item:
                item = item[0]
                item = item.replace(',', '')
                return item
            else:
                return 'Input code is too generic to be matched'

    def __str__(self):
        if len(self.history) == 0:
            return "No queries"
        string_to_return = ''
        self.history.sort()
        for (c,l) in self.history:
            if l[0]:
                lstr = str(l[0][0])
            else:
                lstr = "too generic"
            string_to_return = string_to_return + 'code: ' + str(c) +\
                ' , result: ' + lstr + '\n'
        return string_to_return
