
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer



X_train = np.array(["for(int i = 0; i < 5; i++){}",
                    "for row in data.rows():",
                    "public static void main(String[] args){",
                    "if testVar == 5:",
                    "if(testingVar = 5){return 2*5;}",
                    "elif answers == 7:",
                    "} else {",
                    "else:",
                    "void applyBrakes() {",
                    "def main(argv):",
                    "while(all && i < val.length()){",
                    "while (count < 9):",
                    "int i = 0;",
                    "i = 0"])
y_train = [["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"],["Java"],["Python"]]
X_test = np.array(['int var = 10;',
                   'if alex:',
                   'for i in range(5):'])   
target_names = ['Java', 'Python']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

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




