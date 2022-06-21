from operator import index
import streamlit as st
st.set_page_config(layout="wide")
header_container = st.container()

with header_container:
	st.markdown("<h1 style='text-align: center; color: #1434A4;'>Cooking Time App </h1>", unsafe_allow_html=True)
	st.markdown("<h5 style='text-align: center; color: #2F2E2D;'>Get to know the average cooking time of orders and kitchens!</h5>", unsafe_allow_html=True)	

	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
	st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
	#st.markdown("<h6 style='text-align: center; #2F2E2D;'>Choose the model</h6>", unsafe_allow_html=True)

	choose=st.radio("Choose the model",("Descriptive Analysis", "Linear Regression","Random Forest","ARIMA"))
	st.write("**You have selected:**", choose)

	if choose == 'Descriptive Analysis':

		st.map()
		


	elif(choose  =='Linear Regression'):
		#st.write('You selected Linear Regression.')

		c1, c2 = st.columns((1,1))		

		brand = c1.selectbox('Select the Brand',('Avocalia', 'Bottaniko', 'Brunch & Munch', 'Burritos & Co', 'Cacerola', 'Cafe Amor Perfect', 'La Cuadra', 'Wok Pok', 'Grab & Drink Tienda', 'Tacos & Co'))
#c1.text('hola')

		category = c2.selectbox('Select the Category', ('Plato Fuerte', 'Panadería', 'Postres'))
		st.metric(label="RMSE", value="0.54", delta="5.4")
#c2.text('hola2')


	elif(choose  =='Random Forest'):

		c1, c2 = st.columns((1,1))	
		c1.map()
		max_depth = c2.slider('Select the max depth of the model', min_value=10, max_value=100, value=20, step = 10)
		n_estimators = c2.selectbox('Select numbers of trees', options = [100, 200 ,300], index=0)
		input_feature = c2.text_input('Select features for the model', 'brand')
		#c2.subheader('MAE of the model is...')
		c2.metric(label="MAE (Mean Absolute Error)", value="0.54")
		c2.metric(label="R (R squared Score)", value="0.75")
	else:
		st.write("You selected ARIMA.")
    

