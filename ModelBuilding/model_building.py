
from sklearn.model_selection import train_test_split

X= df_encoded.drop(['output'], axis = 1)
y = df_encoded['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 42)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logreg =  LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation


scores = cross_val_score(logreg, X, y, cv=kf, scoring='accuracy')


print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification report:")
print(classification_report(y_test, y_pred))

