from operator import index
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.figure_factory as ff
import os
import psycopg2
from datetime import datetime, timedelta
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import chi2_contingency
import warnings
#mport seaborn as sns
#from tqdm import tqdm
from scipy.stats import shapiro 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
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
	df = df.drop(labels=['order_id', 'provider_orderid', 'kitchenid'], axis=1)#.head(10000)
	return df.head(10000)

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
	df2 = pd.read_csv(path, engine='python')
	df2.reset_index()
	df2.columns = ['polygon_name', 'polygon', 'point']
	return df2



def descriptive():
	#pio.templates
	df = get_data()
	zone = st.selectbox('Select the Zone',['All zones' , 'Usaquen', 'Suba', 'Salitre', 'Andes', 'Castellana', 'Colina', 'Engativá', 'Chapinero'])
			
	c1, c2 = st.columns((1,1))

	if zone == 'All zones':
		
		fig = px.box(df, x="polygon_name", y='minutes')
		fig2 = px.scatter(df, x='total', y='minutes', trendline="ols", trendline_color_override="red").update_traces(marker=dict(color='#2FC584'))
		fig3 = px.scatter(df, x='num_items', y='minutes').update_traces(marker=dict(color='#17becf'))
		fig4 = px.scatter(df, x='logcreatedat', y="minutes").update_traces(marker=dict(color='#F19A68'))

		subset = pd.crosstab(df["brand"], df['polygon_name'])
		p = chi2_contingency(subset)[1]
		def df_to_plotly(subset):
			return {'z': subset.values.tolist(),
		    'x': subset.columns.tolist(),
            'y': subset.index.tolist()}
		fig5 = go.Figure(data=go.Heatmap(df_to_plotly(subset), colorscale = 'peach'))

		df5 = df[['minutes', 'num_items', 'total']]
		df_corr = df5.corr() 
		x = list(df_corr.columns)
		y = list(df_corr.index)
		z = np.array(df_corr)

		fig6 = ff.create_annotated_heatmap(
			z,
			x = x,
			y = y ,
			annotation_text = np.around(z, decimals=2),
			hoverinfo='z',
			colorscale='blues')

		fig.update_layout(title_text='<b>Cooking time by Zone<b>',
                          title_x=0.5, xaxis_title='Zone',
                          yaxis_title='Time  (minutes)', #template = 'seaborn',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')
		)
						  
		
		fig2.update_layout(title_text='<b>Cooking Time and Cost of the Order<b>',
                          title_x=0.5, xaxis_title='Cost of the Order',
                          yaxis_title='Time  (minutes)'
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)'
		)
						  
		fig3.update_layout(title_text='<b>Cooking Time and Number of Items per Order<b>',
                          title_x=0.5, xaxis_title='Number of Items per Order',
                          yaxis_title='Time  (minutes)', #template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')
		)
		
		fig4.update_layout(title_text='<b>Cooking time by Date<b>',
                          title_x=0.5, xaxis_title='Date',
                          yaxis_title='Time  (minutes)', #template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')
		)

		fig5.update_layout(title_text='<b>Cooking Time by Brand and Zone *',
                          title_x=0.5, xaxis_title='Zone',
                          yaxis_title='Brand', template = 'simple_white'
						  #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')	
		)

		fig6.update_layout(title_text='<b>Correlation Matrix*<b>', title_x=0.5)
          

		c1.plotly_chart(fig, use_container_width=True)
		c2.plotly_chart(fig2, use_container_width=True)
		c1.plotly_chart(fig3, use_container_width=True)
		c2.plotly_chart(fig4, use_container_width=True)
		c1.plotly_chart(fig6, use_container_width=True)
		c2.plotly_chart(fig5, use_container_width=True)
		c2.markdown('***Chi Square test (p value of 0.0) suggests that the Brand and Zone have a statistically significant relationship**')
		c1.markdown("***Considering all zones  the Pearson's correlation between total cost of the order and number of items is weak (r = 0.4)**")
		c1.markdown("***The Pearson's correlation between total cost of the order and the cooking time in minutes is weak (r = 0.2)**")

	if zone != 'All zones' and zone != '':

		filter = df[df['polygon_name']==zone]
		fig = px.box(df, x=filter['polygon_name'], y=filter['minutes'])
		fig2 = px.scatter(df, x=filter['total'], y=filter['minutes'], trendline="ols", trendline_color_override="red").update_traces(marker=dict(color='#2FC584'))
		fig3 = px.scatter(df, x=filter['num_items'], y=filter['minutes']).update_traces(marker=dict(color='#17becf'))
		fig4 = px.scatter(df, x=filter['logcreatedat'], y=filter['minutes']).update_traces(marker=dict(color='#F19A68'))

		subset = pd.crosstab(df["brand"], filter['polygon_name'])
		p = chi2_contingency(subset)[1]
		def df_to_plotly(subset):
			return {'z': subset.values.tolist(),
		    'x': subset.columns.tolist(),
            'y': subset.index.tolist()}
		fig5 = go.Figure(data=go.Heatmap(df_to_plotly(subset)))

		df5 = filter[['minutes', 'num_items', 'total']]
		df_corr = df5.corr() 
		x = list(df_corr.columns)
		y = list(df_corr.index)
		z = np.array(df_corr)

		fig6 = ff.create_annotated_heatmap(
			z,
			x = x,
			y = y ,
			annotation_text = np.around(z, decimals=2),
			hoverinfo='z',
			colorscale='blues')


		fig.update_layout(title_text='<b>Cooking time by Zone<b>',
                          title_x=0.5, xaxis_title='Zone',
                          yaxis_title='Time  (minutes)', #template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')
		)
						  
		
		fig2.update_layout(title_text='<b>Cooking Time and Cost of the Order<b>',
                          title_x=0.5, xaxis_title='Cost of the Order',
                          yaxis_title='Time  (minutes)', #template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)'
		)
						  
		fig3.update_layout(title_text='<b>Cooking Time and Number of Items per Order<b>',
                          title_x=0.5, xaxis_title='Number of Items per Order',
                          yaxis_title='Time  (minutes)', #template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')
		)
		
		fig4.update_layout(title_text='<b>Cooking time by Date<b>',
                          title_x=0.5, xaxis_title='Date',
                          yaxis_title='Time  (minutes)', #template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')
		)

		fig5.update_layout(title_text='<b>Cooking Time by Brand and Zone',
                          title_x=0.5, xaxis_title='Zone',
                          yaxis_title='Brand', template = 'simple_white',
                          #paper_bgcolor='rgba(0,0,0,0)',
                          #plot_bgcolor='rgba(0,0,0,0)')	
		)

		fig6.update_layout(title_text='<b>Correlation Matrix<b>')

		c1.plotly_chart(fig, use_container_width=True)
		c2.plotly_chart(fig2, use_container_width=True)
		c1.plotly_chart(fig3, use_container_width=True)
		c2.plotly_chart(fig4, use_container_width=True)
		c1.plotly_chart(fig6, use_container_width=True)
		c2.plotly_chart(fig5, use_container_width=True)
		#c2.markdown('**Chi Square test (p value of 0.0) suggests that the Brand and Zone have a statistically significant relationship**')
	
	#st.text(df2.head(10))
	#c1.text(p)
	#c2.text("p-value of Chi-square test for Brand vs. Kitchen=", p)
	#c1.markdown("<h1 style='text-align: center>**p-value of Chi-square test for Brand vs. Kitchen=0.0** </h1>", unsafe_allow_html=True)

def regression():
	#st.write('You selected Linear Regression.')
	df2 = get_data()
	np.random.seed(1337)             
	ndata = len(df2)
	# Randomly choose 0.8n indices between 1 and n
	idx_train = np.random.choice(range(ndata),int(0.8*ndata),replace=False)
	# The test set is comprised from all the indices that were
	# not selected in the training set:
	idx_test  = np.asarray(list(set(range(ndata)) - set(idx_train)))
	train3     = df2.iloc[idx_train] # the training data set
	test3      = df2.iloc[idx_test]  # the test data set

	model2 = smf.ols(formula = "minutes ~ kitchen+brand+provider_name+check_orders+total+polygon_name", data = train3).fit()
	test3['prediccion'] =  model2.predict(test3)
	r2 = round(model2.rsquared_adj, 2)
	aic = round(model2.aic, 2)
	bic = round(model2.bic, 2)

	def MAE(prediction,true_values):
		return np.mean(                                                      # Mean
                np.abs(                                                  # Absolute
                        prediction-true_values                            # Error
                    )
                )
	MAE = round(MAE(model2.predict(test3) ,test3.minutes), 2)

	def RMSE(prediction,true_values):
		return np.sqrt(                                                          # Root
            np.mean(                                                      # Mean
                np.square(                                                # Squared
                         prediction-true_values                           # Error
                )
            )
        )
	RMSE = round(RMSE(model2.predict(test3) ,test3.minutes), 2)

	Salitre = round(test3[test3['polygon_name']=='Salitre']['prediccion'].mean(), 2)
	Castellana = round(test3[test3['polygon_name']=='Castellana']['prediccion'].mean(), 2)
	Chapinero = round(test3[test3['polygon_name']=='Chapinero']['prediccion'].mean(), 2)
	Colina = round(test3[test3['polygon_name']=='Colina']['prediccion'].mean(), 2)
	Engativa = round(test3[test3['polygon_name']=='Engativa']['prediccion'].mean(), 2)
	Suba = round(test3[test3['polygon_name']=='Suba']['prediccion'].mean(), 2)
	Usaquen = round(test3[test3['polygon_name']=='Usaquen']['prediccion'].mean(), 2)
	Andes = round(test3[test3['polygon_name']=='Andes']['prediccion'].mean(), 2)

	c1, c2, c3, c4 = st.columns((3, 1, 1, 1))	
	c1.map()
	c2.metric(label="RMSE (Root Mean Squared Error)", value=RMSE)
	c2.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
	asumptions = c2.radio("Check the asumptions of the model*",("No", "Yes"))#, horizontal=False)
	c3.metric(label="R2 (R squared Score)", value=r2)
	c4.metric(label="MAE (Mean Absolute Error)", value=MAE)
	c3.metric(label="AIC", value=aic)
	c4.metric(label="BIC", value=bic)
	c2.text("")
	c2.text("")
	c2.text("")
	c3.metric(label="", value="")
	c4.metric(label="", value="")
	
	
		
	#c1, c2 = st.columns((1, 1))
	zone2 = c2.selectbox('Select the Zone 1',('Suba', 'Usaquen', 'Salitre', 'Andes', 'Castellana', 'Colina', 'Engativá', 'Chapinero'))
	zone3 = c3.selectbox('Select the Zone 2',('Usaquen', 'Suba', 'Salitre', 'Andes', 'Castellana', 'Colina', 'Engativá', 'Chapinero'))
	zone4 = c4.selectbox('Select the Zone 3',('Salitre', 'Usaquen', 'Suba', 'Andes', 'Castellana', 'Colina', 'Engativá', 'Chapinero'))

	Beta_Usaquen = abs(round(model2.params['polygon_name[T.Usaquen]'],2))
	Beta_Salitre = abs(round(model2.params['polygon_name[T.Salitre]'],2))
	Beta_Castellana = round(model2.params['polygon_name[T.Castellana]'],2)
	Beta_Chapinero = round(model2.params['polygon_name[T.Chapinero]'],2)
	Beta_Colina = abs(round(model2.params['polygon_name[T.Colina]'], 2))
	Beta_Engativa = round(model2.params['polygon_name[T.Engativa]'],2)
	Beta_Suba = round(model2.params['polygon_name[T.Suba]'],2)
		
	#c2.text(model2.summary())	
	if zone2 == 'Usaquen':
		c2.subheader('The cooking time for Usaquen is '+str(Beta_Usaquen)+' minutes lower than Andes')
	elif zone2 == 'Suba':
		c2.subheader('The cooking time for Suba is '+str(Beta_Suba)+' minutes higher than Andes')
	elif zone2 == 'Salitre':
		c2.subheader('The cooking time for Salitre is '+str(Beta_Salitre)+' minutes lower than Andes')
	elif zone2 == 'Andes':
		c2.subheader('The cooking time for Andes is '+str(Andes)+' minutes')
	elif zone2 == 'Castellana':
		c2.subheader('The cooking time for Castellana is '+str(Beta_Castellana )+' minutes higher than Andes')
	elif zone2 == 'Colina':
		c2.subheader('The cooking time for Colina is '+str(Beta_Colina )+' minutes lower than Andes')
	elif zone2 == 'Engativá':
		c2.subheader('The cooking time for Engativá is '+str(Beta_Engativa )+' minutes higher than Andes')
	else:
		c2.text('The cooking time for Chapinero is '+str(Beta_Chapinero)+' minutes higher than Andes')
	
	if zone3 == 'Usaquen':
		c3.subheader('The cooking time for Usaquen is '+str(Beta_Usaquen)+' minutes lower than Andes')
	elif zone3 == 'Suba':
		c3.subheader('The cooking time for Suba is '+str(Beta_Suba)+' minutes higher than Andes')
	elif zone3 == 'Salitre':
		c3.subheader('The cooking time for Salitre is '+str(Beta_Salitre)+' minutes lower than Andes')
	elif zone3 == 'Andes':
		c3.subheader('The cooking time for Andes is '+str(Andes)+' minutes')
	elif zone3 == 'Castellana':
		c3.subheader('The cooking time for Castellana is '+str(Beta_Castellana )+' minutes higher than Andes')
	elif zone3 == 'Colina':
		c3.subheader('The cooking time for Colina is '+str(Beta_Colina )+' minutes lower than Andes')
	elif zone3 == 'Engativá':
		c3.subheader('The cooking time for Engativá is '+str(Beta_Engativa )+' minutes higher than Andes')
	else:
		c3.subheader('The cooking time for Chapinero is '+str(Beta_Chapinero)+' minutes higher than Andes')

	if zone4 == 'Usaquen':
		c4.subheader('The cooking time for Usaquen is '+str(Beta_Usaquen)+' minutes lower than Andes')
	elif zone4 == 'Suba':
		c4.subheader('The cooking time for Suba is '+str(Beta_Suba)+' minutes higher than Andes')
	elif zone4 == 'Salitre':
		c4.subheader('The cooking time for Salitre is '+str(Beta_Salitre)+' minutes lower than Andes')
	elif zone4 == 'Andes':
		c4.subheader('The cooking time for Andes is '+str(Andes)+' minutes')
	elif zone4 == 'Castellana':
		c4.subheader('The cooking time for Castellana is '+str(Beta_Castellana )+' minutes higher than Andes')
	elif zone4 == 'Colina':
		c4.subheader('The cooking time for Colina is '+str(Beta_Colina )+' minutes lower than Andes')
	elif zone4 == 'Engativá':
		c4.subheader('The cooking time for Engativá is '+str(Beta_Engativa )+' minutes higher than Andes')
	else:
		c4.subheader('The cooking time for Chapinero is '+str(Beta_Chapinero)+' minutes higher than Andes')

	if asumptions == 'Yes':
		c1, c2, c3, c4 = st.columns((1,1,1,1))
		shapiro2 = shapiro(model2.resid)[1]
		breusch_pagan = sms.het_breuschpagan(model2.resid, model2.model.exog)
		durbin = round(durbin_watson(model2.resid),2) 
		#c1.metric(label="", value='')
		#c2.metric(label="", value='')
		#c3.metric(label="", value='')
		#c4.metric(label="", value='')
		mystyle = '''
		<style>
        p {
			text-align: justify;
        }
    	</style>
    	'''
		st.markdown(mystyle, unsafe_allow_html=True)
		c1.metric(label="Normality Shapiro-Wilk Test p value:", value=shapiro2 )
		c2.metric(label="Heteroscedasticity Breusch-Pagan Test p value:", value=round(breusch_pagan[1],2))
		c3.metric(label="Autocorrelation Durbin-Watson Test statistic:", value=durbin )
		c4.info('***Conclusion: The assumptions (normality and homocedasticity) are not met. The model is not valid for inference.**')
		#c4.markdown("<h6 style='text-align: center; color: ##0a0a0a;'>Conclusion: The assumptions (normality and homocedasticity) are not met. The model is not valid for inference!</h6>", unsafe_allow_html=True)	
		#c4.error('Conclusion: The assumptions (normality and homocedasticity) are not met. The model is not valid for inference')
		st.balloons()
		
		
		

	
	

def randomF():
	df2 = get_data()
	df2 = pd.get_dummies(df2, prefix =['kitchen', 'type', 'brand', 'provider_name', 'polygon_name'])
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
	c2.metric(label="R2 (R squared Score)", value=r2)
	c2.metric(label="MAE (Mean Absolute Error)", value=mae)

	Salitre = X_test[X_test['polygon_name_Salitre']==1]['prediccion'].mean()
	Castellana = X_test[X_test['polygon_name_Castellana']==1]['prediccion'].mean()
	Chapinero = X_test[X_test['polygon_name_Chapinero']==1]['prediccion'].mean()
	Colina = X_test[X_test['polygon_name_Colina']==1]['prediccion'].mean()
	Engativa = X_test[X_test['polygon_name_Engativa']==1]['prediccion'].mean()
	Suba = X_test[X_test['polygon_name_Suba']==1]['prediccion'].mean()
	Usaquen = X_test[X_test['polygon_name_Usaquen']==1]['prediccion'].mean()
	Andes = X_test[X_test['polygon_name_Andes']==1]['prediccion'].mean()
	predicciones = [Chapinero, Castellana, Colina, Salitre, Suba, Andes, Usaquen]

	df3 = get_data2()
	df3.insert(loc=3, column='prediction', value=predicciones)


	


with header_container:
	#my_bar = st.progress(0)
	#for percent_complete in range(100):
		#time.sleep(0.1)
		#my_bar.progress(percent_complete + 1)

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



		
    

