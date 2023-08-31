from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve


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

#better to use confusion matrix for classifiers
y_train_pred = cross_val_predict(sdg_clf, X_train, y_train_5, cv=3) #performs k-fold prediction like cross_val_score, but returns predictions instead of scores
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

#f1 combines recall and precision
print(f1_score(y_train_5, y_train_pred))

#increasing percision, lowers recall and vice versa
y_scores = cross_val_predict(sdg_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plt.plot(thresholds, precisions[:-1], "b--", label = "Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label='Recall', linewidth=2)
plt.vlines(thresholds, 0, 1.0, "k", "dotted", label='Threshold')
plt.show()