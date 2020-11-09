# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')


X = dataset.iloc[:, :3]

y = dataset.iloc[:, -1]




from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))  #experience, test_score, Interview Score