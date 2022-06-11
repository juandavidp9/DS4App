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
	#if choose == 'Linear Regression':
		#st.write('You selected Linear Regression.')
	#elif(choose  =='Random Forest'):
		#st.write('You selected Random Forest.')
	#else:
		#st.write("You selected ARIMA.")
    

c1, c2 = st.columns((1,1))		

c1.option = st.selectbox(
     'Select the Brand',
     ('Avocalia', 'Bottaniko', 'Brunch & Munch', 'Burritos & Co', 'Cacerola', 'Cafe Amor Perfect', 'La Cuadra', 'Wok Pok', 'Grab & Drink Tienda', 'Tacos & Co'))


c2.option = st.selectbox(
     'Select the Category',
     ('Plato Fuerte', 'Panadería', 'Postres'))

st.metric(label="RMSE", value="0.54", delta="5.4")

st.map()
	 
