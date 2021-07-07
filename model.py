#Importing Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


#Loading Dataset
data = pd.read_csv('Iris.csv')
print(data.head)

#Selecting Independent and Dependent variables

# Independent variables
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# Dependent variable
y = data['Species']

#Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Instantiate the model
classifier = RandomForestClassifier()

#Fitting the model
classifier.fit(X_train, y_train)

#Pickle file of model
pickle.dump(classifier, open("model.pkl", "wb" ))


