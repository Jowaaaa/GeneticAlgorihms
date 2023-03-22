from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = datasets.load_breast_cancer

dataset = pd.read_csv('breast cancer.csv')

#label encoder om M & B naar 1 & 0 te veranderen
LE = LabelEncoder()
dataset.iloc[:,1]=LE.fit_transform(dataset.iloc[:,1].values)

#dependent X & independent Y
X = dataset.iloc[:,2:32].values
Y = dataset.iloc[:,1].values

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

clf = MLPClassifier(random_state=1, max_iter=500).fit(X_train, Y_train)
clf.predict_proba(X_test[:1])

clf.predict(X_test[:5, :])

print(clf.score(X_test, Y_test))




#print(dataset.head())
#print(dataset.describe())
#print(dataset['diagnosis'].value_counts())