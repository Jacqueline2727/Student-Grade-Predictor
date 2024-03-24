import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";") #change the delimiter

#Trim the data to what we need
data = data[["G1", "G2","G3", "studytime","failures","absences","schoolsup","higher"]]
data["schoolsup"] = data["schoolsup"].map({'yes':1, 'no':0})
data["higher"]=data["higher"].map({'yes':1,'no':0})
print(data.head()) #first 5 instances

predict = "G3" #G3 is the label - what we are looking for

x = np.array(data.drop(predict, axis=1))  # Correctly dropping the G3
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#split the data training and test sets - 10% of the data will be tested

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train) #fit the data to find the best fit
acc = linear.score(x_test, y_test) #returns accuracy of our model

print("Coefficients: ",linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])

print("Accuracy: ",acc)