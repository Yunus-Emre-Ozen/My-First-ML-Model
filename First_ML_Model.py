import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


data = pd.read_csv(r"https://raw.githubusercontent.com/dphi-official/Datasets/master/HR_comma_sep.csv")

#Look into properties of data
data.info()
data.describe()
data.duplicated()

#We should drop duplicates for our ML model work properly. 
filtered_data = data.drop_duplicates()

# Categorical columns
cat_col = [col for col in filtered_data.columns if filtered_data[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in filtered_data.columns if filtered_data[col].dtype != 'object']
print('Numerical columns :',num_col)


#We should use ordinal encoding for salary

filtered_data["salary"] = filtered_data["salary"].replace(['low', 'medium', 'high'],[0,1,2])


#Then we should use one hot encoding for department column
one_hot_encoded_filtered_data = pd.get_dummies(filtered_data, columns = ['Department'])


#Then we use final data 

final_data = one_hot_encoded_filtered_data


X = final_data.drop("left", axis = 1)
Y = final_data["left"]


# import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state = 1)


log_model = LogisticRegression()
# Fit the model
log_model.fit(X_train, Y_train)


#After logistic regression we would look into predictions

predictions = log_model.predict(X_test)

#Then in the end we look how accurate our model
from sklearn.metrics import accuracy_score

accuracy_score(Y_test, predictions)
print(accuracy_score(Y_test, predictions))

#Then we should use confussion matrix to examine our model better

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(Y_test, predictions)
print(confusion_matrix(Y_test, predictions))


#In the end lets look into how we visualize to confusion matrix, we work under a big data set so I use percentages

sns.heatmap(matrix/np.sum(matrix), annot=True, cmap = "Blues")

