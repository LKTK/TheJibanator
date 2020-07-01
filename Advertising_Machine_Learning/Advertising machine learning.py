import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Carregando o Dataset com pandas
arquivo= pd.read_csv('F:\TCC_Lujan\TheJibanator\Advertising_Machine_Learning\Advertising.csv',index_col=0)
arquivo.shape

# Analise dos dados de propagando com o seaborn
sns.pairplot(arquivo,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height= 7,aspect=0.7, kind='reg')

feature_cols=['TV','Radio']

#separando o conjunto de dados
x= arquivo[['TV','Radio','Newspaper']]
y=arquivo['Sales']
x_train,x_test,y_train,y_test =train_test_split(x,y, test_size=0.3)


# aplicando o modelo
linreg = LinearRegression()
linreg.fit(x_train,y_train)

print('ponto em q intercepta o eixo y: ',linreg.intercept_) #coeficiente da equação de regressao linear
print('Coeficientes da equação de regressão linear: ',linreg.coef_) #coeficiente da equação de regressão linear


#Models evaluation metrics for regression
true= [100,50,30,20]
pred= [90,50,50,30]
print('-------------------------')
#Mean Absolute Erro (MAE)
print('MAE calculado a mão:',(10+0+20+10)/4)
#calculate MAE using scikit-learn
print('MAE calculado pelo scikit-learn:',metrics.mean_absolute_error(true,pred))
print('-------------------------')
#Calculate Mean Squared Error (MSE)
print('MSE calculado a mão:',(10**2+0**2+20**2+10**2)/4)
#Calculate MSE using scikit-learn
print('MSE calculado pelo scikit-learn:',metrics.mean_squared_error(true,pred))
print('-------------------------')
#Calculate Root Mean Saquared Error (RMSE)
print('RMSE calculado a mão:',np.sqrt((10**2+0**2+20**2+10**2)/4))
#Calculate RMSE using scikit-learn
print('RMSE calculado pelo scikit-learn:',np.sqrt(metrics.mean_squared_error(true,pred)))
print('-------------------------')

#Para o modelo de advertising
y_pred= linreg.predict(x_test)
print('Calculo de Erro RMSE do modelo: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('-------------------------')
#Analisando que o parametro Newspaper não influi muito na propaganda

#create a Python list of feature names
feature_cols=['TV','Radio']
#use the list to select a subset of the original Dataframe
x= arquivo[feature_cols]
#select a Series from DataFrame
y= arquivo['Sales']
#split into training and testing sets
x_train,x_test,y_train,y_test =train_test_split(x,y, test_size=0.3, random_state=1)
# fit the model to the training data (learn the coefficients)
linreg.fit(x_train,y_train)
#make predictions on the testing set
y_pred= linreg.predict(x_test)
#compute RMSE of our predictions 
print('Calculo do Erro RMSE do modelo sem o parametro Newspaper: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))




