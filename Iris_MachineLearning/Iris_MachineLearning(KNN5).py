import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# IMPORTANDO O BANCO DE DADOS
arquivo= pd.read_csv('F:\\TCC_Lujan\Projetos Pessoais\\The Jibanator\\Iris_MachineLearning\\datasets_19_420_Iris.csv')

arquivo['Species']= arquivo['Species'].replace('Iris-setosa',0)
arquivo['Species']= arquivo['Species'].replace('Iris-versicolor',1)
arquivo['Species']= arquivo['Species'].replace('Iris-virginica',2)


x= arquivo[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=arquivo['Species']

# SEPARANDO EM CONJUNTO DE TREINO E DE TESTES 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# MODELO K NEAREST NEIGHBORS PARA K = 5
print('MODELO K NEAREST NEIGHBORS PARA K = 5','\n')
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred= knn.predict(x_test)
print('precisão de acertos no conjunto teste: ',accuracy_score(y_test,y_pred))
#Analise dos resultados
print('teste de previsão')
info= x_test.index[10:15]    #<---- valores para ser previstos, podem ser alterados dentro do range do vetor X_teste
previsao_teste= knn.predict(x_test[10:15]) #<---- valores para ser previstos, podem ser alterados dentro do range do vetor X_teste
#print(previsao_teste)
testeDIC={'valor real':[arquivo.loc[info[0],'Species'],arquivo.loc[info[1],'Species'],arquivo.loc[info[2],'Species'],arquivo.loc[info[3],'Species'],arquivo.loc[info[4],'Species']],
        'valor previsto':[previsao_teste[0],previsao_teste[1],previsao_teste[2],previsao_teste[3],previsao_teste[4]]}
testeDF= pd.DataFrame(testeDIC)
print(testeDF)
print('--------------------------------------------------------------')

    

#predict=([2,4,3,1],[5,4,3,2])
#x=predict.reshape(-1,4) para previsão solo precisa usar o reshape junto do numpy array
#knn.predict(predict)











    
    








