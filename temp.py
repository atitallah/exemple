# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import panda as pd
dataset = pd.read_csv('Salary_Data.csv')
#pour lire un fichier xl 
x = dataset.iloc[:, :-1].values
#importer les nombres d'annees jusqu'a apres la dernier année 
y = dataset.iloc[:, -1].values
#importer les salaires 
from sklearn.model_selection import train_test_split
#fonction pour trainer 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
#diviser les x et y partie pour le trainer et partie pour le test 
from sklearn.linear_model import LinearRegression
#fonction pour tracer la ligne
regressor = LinearRegression()
#?
regressor.fit(x_train, y_train)
#?
y_pred = regressor.predict(x_test)
#?
plt.scatter(x_train, y_train, color = 'red')
#tracer la ligne par la couleur rouge 
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
#marquer les points par la couleur bleu
plt.title('Salary vs Experience (Training set)')
#titre 
plt.xlabel('Years of Experience')
#nom de l'axe x
plt.ylabel('Salary')
#nom de l'axe y
plt.show()
#affichage 
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
y_pred2= regressor.predict([[12]])
print(y_pred2)
print("----------------------")
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test)) 



