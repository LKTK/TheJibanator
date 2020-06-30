import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
arquivo= pd.read_csv('F:\\TCC_Lujan\\Projetos Pessoais\\The Jibanator\\vinhos\\datasets_16721_22034_wine_dataset.csv') #caminho para o arquivo de dados

arquivo['style']= arquivo['style'].replace('red',0)
arquivo['style']= arquivo['style'].replace('white',1)

# Separando o conjunto de dados em parametros e variavél alvo
y= arquivo['style']
X= arquivo.drop('style',axis=1)

# Separando conjunto de treino e conjunto de teste
X_treino,X_teste,y_treino,y_teste =train_test_split(X,y, test_size=0.3)

#Criação do modelo

modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(X_treino,y_treino)

#Analise dos resultados
resultado = modelo.score(X_teste,y_teste)
print('Acurácia',resultado)
info= X_teste.index[200:205]    #<---- valores para ser previstos, podem ser alterados dentro do range do vetor X_teste
previsao_teste= modelo.predict(X_teste[200:205]) #<---- valores para ser previstos, podem ser alterados dentro do range do vetor X_teste
#print(previsao_teste)
testeDIC={'valor real':[arquivo.loc[info[0],'style'],arquivo.loc[info[1],'style'],arquivo.loc[info[2],'style'],arquivo.loc[info[3],'style'],arquivo.loc[info[4],'style']],
        'valor previsto':[previsao_teste[0],previsao_teste[1],previsao_teste[2],previsao_teste[3],previsao_teste[4]]}
testeDF= pd.DataFrame(testeDIC)
print(testeDF)