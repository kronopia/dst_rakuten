

### projet Titanic 
# Sprint 9 Data Engineering - 1 Streamlit - 02 Cas Pratique.docx


# Streamlit : cliquer sur "Always re-run" ( pr rafraichir automatiquet )


import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import confusion_matrix

import random
@st.cache_data



# rq : Mémoire cache
# Pour éviter de perdre du temps à chaque mise à jour de la page, nous pouvons utiliser le décorateur @st.cache_data. Il permet de garder en mémoire une valeur de telle sorte que si nous rafraichissons la page Streamlit avec un "Re-run" nous obtenons toujours la même chose.
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)
# Les valeurs de a et b, censées être aléatoires, restent toujours les mêmes.




# Ajouter un titre au Streamlit et créer 3 pages appelées "Exploration", "DataVizualization" et "Modélisation"
st.title( "Projet de classification binaire Titanic" )
st.sidebar.title( "Sommaire" )
pages=[ "Exploration", "DataVizualization", "Modélisation" ]
page=st.sidebar.radio( "Aller vers", pages )


# Créer un dataframe appelé df permettant de lire le fichier train.csv
df=pd.read_csv("train.csv")


### 1. Exploration

# Ecrire "Introduction" en haut de la première page 
if page == pages[0] : 
	st.write( "### Introduction" )

	# Afficher les 10 premières lignes du dataframe df 
	st.dataframe( df.head(10) )

	# Afficher des informations sur le dataframe 
	st.write( df.shape )
	st.dataframe( df.describe() )

	# Créer une checkbox pour choisir d'afficher ou non le nombre de valeurs manquantes 
	if st.checkbox("Afficher les NA") :
	  st.dataframe( df.isna().sum() )



### 2. DataVizualization

# Ecrire "DataVizualization" en haut de la deuxième page 
if page == pages[1] : 
	st.write("### DataVizualization")

	# La variable cible "Survived" prend 2 modalités : 0 si l'individu n'a pas survécu et 1 si l'individu a survécu
	# Afficher dans un plot la distribution de la variable cible
	fig = plt.figure()
	sns.countplot(x = 'Survived', data = df)
	st.pyplot(fig)


	# Analyse descriptive des données pour obtenir le profil type d'un passager du Titanic 
	# Afficher des plots permettant de décrire les passagers du Titanic
	fig = plt.figure()
	sns.countplot(x = 'Sex', data = df)
	plt.title("Répartition du genre des passagers")
	st.pyplot(fig)

	fig = plt.figure()
	sns.countplot(x = 'Pclass', data = df)
	plt.title("Répartition des classes des passagers")
	st.pyplot(fig)

	fig = sns.displot(x = 'Age', data = df)
	plt.title("Distribution de l'âge des passagers")
	st.pyplot(fig)


	# Analyser l'impact des différents facteurs sur la survie ou non des passagers
	# Afficher un countplot de la variable cible en fonction du genre
	fig = plt.figure()
	sns.countplot(x = 'Survived', hue='Sex', data = df)
	st.pyplot(fig)

	# Afficher un plot de la variable cible en fonction des classes
	fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
	st.pyplot(fig)

	# Afficher un plot de la variable cible en fonction des âges
	fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
	st.pyplot(fig)


	# Pour terminer l'analyse multivariée, afficher la matrice de corrélation des variables explicatives
	fig, ax = plt.subplots()
	sns.heatmap(df.select_dtypes(include=[np.number]).corr(), ax=ax)
	st.write(fig)



### 3. Modélisation

if page == pages [2] :
	# Ecrire "Modélisation" en haut de la troisième page
	st.write("### Modélisation")

	# Nous faisons de la classification binaire pour prédire si un passager survit ou non au naufrage du Titanic.
	# Preprocessing du dataframe

	# Supprimer variables non-pertinentes: PassengerID, Name, Ticket, Cabin
	df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

	# Variable target
	y = df[ 'Survived']
	# Variables explicatives catégorielles 
	X_cat = df[['Pclass', 'Sex',  'Embarked']]
	# Variables explicatives numériques
	X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
	
	# Remplacer les valeurs manquantes des variables catégorielles par le mode 
	for col in X_cat.columns:
		X_cat [col] = X_cat[col].fillna(X_cat[col].mode() [0])

	# Remplacer les valeurs manquantes des variables numériques par la médiane
	for col in X_num.columns:
		X_num[col] = X_num[col].fillna(X_num[col].median())

	# Encoder les variables catégorielles
	X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
	

	# Concaténer les variables explicatives encodées et sans valeurs manquantes pour obtenir un dataframe X clean.
	X = pd.concat([X_cat_scaled, X_num], axis = 1)


	# Séparer les données en un ensemble d'entrainement et un ensemble test	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

	# Standardiser les valeurs numériques
	scaler = StandardScaler()
	X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
	X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])


	# Créer une fonction appelée prediction qui prend en argument le nom d'un classifieur et renvoie le classifieur entrainé
	def prediction (classifier):
		if classifier == 'Random Forest':
			clf = RandomForestClassifier()
		elif classifier == 'SVC':
			clf = SVC()
		elif classifier == 'Logistic Regression':
			clf = LogisticRegression()
		clf.fit(X_train, y_train)
		return clf

	def scores(clf, choice):
		if choice == 'Accuracy' :
			return clf.score(X_test, y_test)
		elif choice == 'Confusion matrix':
			return confusion_matrix(y_test, clf.predict(X_test))


	# Creer une "select box" permettant de choisir quel classifieur entrainer.
	choix = ['Random Forest', 'SVC', 'Logistic Regression']
	option = st.selectbox('Choix du modèle', choix)
	st.write('Le modèle choisi est :', option)


	# Entrainer le classifieur choisi en utilisant la fonction prediction précédemment définie et à afficher les résultats.
	clf = prediction(option)
	display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
	if display == 'Accuracy':
	    st.write(scores(clf, display))
	elif display == 'Confusion matrix':
	    st.dataframe(scores(clf, display))






# rq : outil Streamlit Survey : inclure des sondages 
# outil Streamlit Canvas : inclure des dessins 



# Sauvegarde du modèle : 

# Dans le script Python dédié à la modélisation 
# Enregistrement du classifieur clf entrainé (après avoir fait clf.fit()) sous le nom "model" avec Joblib :
# import joblib
# joblib.dump(clf, "model")
# ou 
# Enregistrement du classifieur clf entrainé (après avoir fait clf.fit()) sous le nom "model" avec Pickle :
# import pickle
# pickle.dump(clf, open("model", 'wb'))
# ou 
# Pour des modèles de Deep Learning, il existe le format H5 de Keras.

# Dans le script Python dédié à Streamlit :
# Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Joblib :
# joblib.load("model")
# ou 
# Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Pickle :
# loaded_model = pickle.load(open("model", 'rb'))
# ou 
# DL : load model keras h5 ... 








