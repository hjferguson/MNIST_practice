from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

import matplotlib.pyplot as plt



mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

some_digit = X[0]

#book example to just train off of the 5s

y_train_5 = (y_train == '5') #true for all 5s, false for everything else
y_test_5 = (y_test == '5')

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(X_train, y_train_5)

#performance measures
#cross_val_score(sdg_clf, X_train, y_train_5, cv=3, scoring='accuracy')
#prints [0.95035 0.96035 0.9604 ]

#textbook wants to throw a wrench in things. since accuracy too high off the bat
# dummy_clf = DummyClassifier()
# dummy_clf.fit(X_train, y_train_5)
# print(any(dummy_clf.predict(X_train)))
# #prints false
# print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring='accuracy'))
# #prints: [0.90965 0.90965 0.90965]
# #10% of the data is 5s, so to guess false, the model would be right 90% of the time. 
# #shows why accuracy isn't a good performance marker for classifiers