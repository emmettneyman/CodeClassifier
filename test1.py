
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


X_train = np.array(["for(int x = 0; x < 5; x++){}",
                    "for x in data.rows():",
                    "public static void main(String[] args){",
                    "if x == 5:",
                    "if(x == 5){}",
                    "elif x == 7:",
                    "} else {",
                    "else:",
                    "void applyBrakes() {",
                    "def main(argv):",
                    "while(all && x < val.length()){",
                    "while (x < 5):",
                    "int x = 0;",
                    "x = 0"])

y_train = [["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"]]

X_test = np.array(['int var = 10;',
                   'if emmett:',
                   'if(alex == 4){}else{}'])   

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

# classifier = KNeighborsClassifier(n_neighbors=7, algorithm='auto')
# classifier = DecisionTreeClassifier(max_depth=10)
# classifier = GaussianNB()
# classifier = CountVectorizer()

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

for item in all_labels:
	print(item)


# def main(argv):
# 	print(argv[0])
# 	print(type(argv[0]))

# if __name__ == "__main__":
#    main(sys.argv[1:])




