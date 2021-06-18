import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns # Data Visualization
import sklearn 
from sklearn import preprocessing #For Data Preprocessing, here for encoding categorical data
from sklearn.naive_bayes import GaussianNB # The classifier used
from sklearn.model_selection import train_test_split #Splitting the dataset into training and test data

#preparing the data
data = pd.read_csv('column_3C_weka.csv').dropna()
X = data.iloc[:,0:6]
y = data.iloc[:,-1]

#encoding categorical variables
encoder = preprocessing.LabelEncoder()
data['class'] = encoder.fit_transform(data['class'])

#plotting the data
sns.heatmap(data.corr(), annot=True)

#Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=25)

#Creating the model for prediction
predictor = GaussianNB()

#Fitting the data to the model
predictor.fit(X_train,y_train)

#Printing the results and the accuracy
accuracy = predictor.score(X_test,y_test)
comparison = predictor.predict(X_test)==y_test
final = pd.DataFrame([predictor.predict(X_test),y_test,comp]).T
final.columns = ['Predictions','Labels','Correct?']
final1 = pd.merge(X_test,final,left_index=True,right_index=True)

print('***\tThe accuracy of this model = ',accuracy,'%\t***\n\n')

final1.to_csv('biomech ortho patients.csv',encoding='utf-8',index=False)
