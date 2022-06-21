from operator import index
import streamlit as st
import pandas as pd
import numpy as np
import os
import psycopg2
from datetime import datetime, timedelta
import warnings
#from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
import multiprocessing
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
header_container = st.container()

@st.cache
def get_data():
	#We create the conecction with the database
	conn = psycopg2.connect(host="data-prod.cem7ltlisydy.us-east-1.redshift.amazonaws.com", port = 5439, database="data-prod-co", user="food_user", password="F00do105y2o22")
	cur = conn.cursor()
	cur.execute("""SELECT * FROM cooking_time.orders_cooking_new""")
	orders_cooking_new = cur.fetchall()
	#we get the dataframe
	df = pd.DataFrame(orders_cooking_new)
	df.reset_index()
	df.columns = ['order_id', 'provider_orderid', 'kitchenid', 'kitchen','type','brand','provider_name', 'logcreatedat','check_orders','order_secs', 'num_items', 'total', 'polygon_name']
	df['kitchenid'] = df['kitchenid'].astype('category')
	df['kitchen'] = df['kitchen'].astype('category')
	df['brand'] = df['brand'].astype('category')
	df['num_items'] = df['num_items'].astype('float64')
	df['total'] = df['total'].astype('float64')
	df['provider_name'] = df['provider_name'].str.upper()
	df['minutes'] = df['order_secs']/60 
	df2 = df.copy()
	df2 = df.drop(labels=['order_id', 'provider_orderid', 'kitchenid', 'order_secs', 'logcreatedat'], axis=1)
	df2 = pd.get_dummies(df2, prefix =['kitchen', 'brand', 'provider_name', 'polygon_name'], drop_first=True)
	df2 = df2.dropna()
	return df2

 
def descriptive():
	st.map()

def regression():
	#st.write('You selected Linear Regression.')
	c1, c2 = st.columns((1,1))	
	brand = c1.selectbox('Select the Brand',('Avocalia', 'Bottaniko', 'Brunch & Munch', 'Burritos & Co', 'Cacerola', 'Cafe Amor Perfect', 'La Cuadra', 'Wok Pok', 'Grab & Drink Tienda', 'Tacos & Co'))
	category = c2.selectbox('Select the Category', ('Plato Fuerte', 'Panadería', 'Postres'))
	st.metric(label="RMSE", value="0.54", delta="5.4")

def randomF():
	df2 = get_data()

	c1, c2 = st.columns((1,1))	
	c1.map()
	max_depth2 = c2.slider('Select the max depth of the model', min_value=10, max_value=50, value=20, step = 10)
	n_estimators2 = c2.selectbox('Select numbers of trees', options = [10, 20 ,30, 40], index=0)
	#input_feature = c2.text_input('Select features for the model', 'brand')
	#c2.subheader('MAE of the model is...')
	
	
	X_train, X_test, y_train, y_test = train_test_split(
                                        df2.drop(columns = ["minutes"]),
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

	rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = pred,
        squared = False
       )
	mae = mean_squared_error(y_test, pred)
	r2 = r2_score(y_test, pred)
	c2.metric(label="RMSE (Mean Squared Error)", value=rmse)
	c2.metric(label="R (R squared Score)", value=r2)
	c2.metric(label="Mean Absolute Error", value=mae)
	


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



		
    

