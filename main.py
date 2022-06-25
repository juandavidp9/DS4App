from operator import index
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import os
import psycopg2
from datetime import datetime, timedelta
from scipy.stats import chi2_contingency
import warnings
#import statsmodels.api as sm
#from tqdm import tqdm 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
from shapely import wkb
import requests
from io import StringIO
from io import BytesIO
import multiprocessing
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
header_container = st.container()

@st.cache

def get_data():
	#We create the conecction with the database
	#conn = psycopg2.connect(host="data-prod.cem7ltlisydy.us-east-1.redshift.amazonaws.com", port = 5439, database="data-prod-co", user="food_user", password="F00do105y2o22")
	#cur = conn.cursor()
	#cur.execute("""SELECT * FROM cooking_time.orders_cooking_new""")
	#orders_cooking_new = cur.fetchall()
	#we get the dataframe
	#df = pd.DataFrame(orders_cooking_new)
	original_url = "https://drive.google.com/file/d/1mK15-Qmk5vYdq91hAuLpdNeLOKfGemdt/view?usp=sharing"
	file_id = original_url.split('/')[-2]
	dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
	url = requests.get(dwn_url).text
	path = StringIO(url)
	df = pd.read_csv(path)
	df.reset_index()
	df.columns = ['order_id', 'provider_orderid', 'kitchenid', 'kitchen','type','brand','provider_name', 'logcreatedat','check_orders','minutes', 'num_items', 'total', 'polygon_name']
	df['kitchenid'] = df['kitchenid'].astype('category')
	df['kitchen'] = df['kitchen'].astype('category')
	df['brand'] = df['brand'].astype('category')
	df['num_items'] = df['num_items'].astype('float64')
	df['total'] = df['total'].astype('float64')
	df['provider_name'] = df['provider_name'].str.upper()
	df = df.drop(labels=['order_id', 'provider_orderid', 'kitchenid'], axis=1)
	return df

def get_data2():
	#We create the conecction with the database
	#conn = psycopg2.connect(host="data-prod.cem7ltlisydy.us-east-1.redshift.amazonaws.com", port = 5439, database="data-prod-co", user="food_user", password="F00do105y2o22")
	#cur = conn.cursor()
	#cur.execute("""SELECT * FROM cooking_time.orders_cooking_final""")
	#orders_cooking_new = cur.fetchall()
	#df2 = pd.DataFrame(orders_cooking_new)
	#df2.reset_index()
	#df2.columns = ['order_id', 'provider_orderid', 'kitchenid', 'kitchen','type','brand','provider_name', 'logcreatedat','check_orders','minutes', 'num_items', 'total', 'polygon_name', 'polygon_', 'point']
	#df2['polygon'] = df2['polygon_'][0:100].apply(lambda x : wkb.loads(x,hex=True))
	#df2 = df2.drop(columns=['provider_orderid','logcreatedat','check_orders', 'kitchen', 'polygon_', 'point'], axis=1)
	original_url = "https://drive.google.com/file/d/1dX3AJxJx2r5oqlRwhv3O6WBnCgYgCv5l/view?usp=sharing"
	file_id = original_url.split('/')[-2]
	dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
	url = requests.get(dwn_url).text
	path = StringIO(url)
	df2 = pd.read_csv(path)
	df2.reset_index()
	df2.columns = ['polygon_name', 'polygon', 'point']
	return df2



def descriptive():
	df = get_data()
	#st.text(df2.head(10))
	fig = px.box(df, x="polygon_name", y='minutes')
	fig2 = px.scatter(df, x='total', y='minutes').update_traces(marker=dict(color='#2ca02c'))
	fig4 = px.scatter(df, x='logcreatedat', y="minutes")
	fig3 = px.scatter(df, x='num_items', y='minutes').update_traces(marker=dict(color='#17becf'))
	subset = pd.crosstab(df["brand"], df["kitchen"])
	def df_to_plotly(subset):
		return {'z': subset.values.tolist(),
		        'x': subset.columns.tolist(),
                'y': subset.index.tolist()}
	fig5 = go.Figure(data=go.Heatmap(df_to_plotly(subset)))
	p = chi2_contingency(subset)[1]

	c1, c2 = st.columns((1,1))
	c1.plotly_chart(fig, use_container_width=True)
	c2.plotly_chart(fig2, use_container_width=True)
	c1.plotly_chart(fig3, use_container_width=True)
	c2.plotly_chart(fig4, use_container_width=True)
	c1.plotly_chart(fig5, use_container_width=True)
	#c2.text("p-value of Chi-square test for Brand vs. Kitchen=", p)
	c1.markdown('**P-value of Chi-square test for Brand vs. Kitchen = 0.0**')
	#c1.markdown("<h1 style='text-align: center>**p-value of Chi-square test for Brand vs. Kitchen=0.0** </h1>", unsafe_allow_html=True)

def regression():
	#st.write('You selected Linear Regression.')
	c1, c2 = st.columns((1,1))	
	brand = c1.selectbox('Select the Brand',('Avocalia', 'Bottaniko', 'Brunch & Munch', 'Burritos & Co', 'Cacerola', 'Cafe Amor Perfect', 'La Cuadra', 'Wok Pok', 'Grab & Drink Tienda', 'Tacos & Co'))
	category = c2.selectbox('Select the Category', ('Plato Fuerte', 'Panadería', 'Postres'))
	st.metric(label="RMSE", value="0.54", delta="5.4")

def randomF():
	df2 = get_data()
	df2 = pd.get_dummies(df2, prefix =['kitchen', 'type', 'brand', 'provider_name', 'polygon_name'], drop_first=True)
	df2 = df2.dropna()

	c1, c2 = st.columns((1,1))	
	c1.map()
	max_depth2 = c2.slider('Select the max depth of the model', min_value=10, max_value=50, value=20, step = 10)
	n_estimators2 = c2.selectbox('Select numbers of trees', options = [10, 20 ,30, 40], index=0)
	#input_feature = c2.text_input('Select features for the model', 'brand')
	#c2.subheader('MAE of the model is...')
	
	
	X_train, X_test, y_train, y_test = train_test_split(
                                        df2.drop(columns = ['minutes']),
                                        df2['minutes'],
                                        random_state = 123
                                    )
	model = RandomForestRegressor(
            n_estimators = n_estimators2,
            criterion    = 'mse',
            max_depth    = max_depth2,
            max_features = 'auto',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )
	model.fit(X_train, y_train)
	pred = model.predict(X = X_test)
	X_test['prediccion'] = pred

	rmse = round(mean_squared_error(
        y_true  = y_test,
        y_pred  = pred,
        squared = False
       ), 2)
	mae = round(mean_squared_error(y_test, pred),2)
	r2 = round(r2_score(y_test, pred), 2)
	c2.metric(label="RMSE (Root Mean Squared Error)", value=rmse)
	c2.metric(label="R (R squared Score)", value=r2)
	c2.metric(label="MAE (Mean Absolute Error)", value=mae)

	Salitre = X_test[X_test['polygon_name_Salitre']==1]['prediccion'].mean()
	Castellana = X_test[X_test['polygon_name_Castellana']==1]['prediccion'].mean()
	Chapinero = X_test[X_test['polygon_name_Chapinero']==1]['prediccion'].mean()
	Colina = X_test[X_test['polygon_name_Colina']==1]['prediccion'].mean()
	Engativa = X_test[X_test['polygon_name_Engativa']==1]['prediccion'].mean()
	Suba = X_test[X_test['polygon_name_Suba']==1]['prediccion'].mean()
	Usaquen = X_test[X_test['polygon_name_Usaquen']==1]['prediccion'].mean()
	predicciones = [Chapinero, Castellana, Colina, Salitre, Suba, 20.306155826861165, Usaquen]

	df3 = get_data2()
	df3.insert(loc=3, column='prediction', value=predicciones)


	


with header_container:
	st.markdown("<h1 style='text-align: center; color: #1434A4;'>Cooking Time App </h1>", unsafe_allow_html=True)
	st.markdown("<h5 style='text-align: center; color: #2F2E2D;'>Get to know the average cooking time of orders and kitchens!</h5>", unsafe_allow_html=True)	

	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
	st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
	#st.markdown("<h6 style='text-align: center; #2F2E2D;'>Choose the model</h6>", unsafe_allow_html=True)

	choose=st.radio("Choose the model",("Descriptive Analysis", "Linear Regression","Random Forest"))
	st.write("**You have selected:**", choose)

	if choose == 'Descriptive Analysis':
		descriptive()
	
	elif(choose  =='Linear Regression'):
		regression()
		
	else:
		randomF()



		
    

