import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import f1_score
#Streamlit part

st.set_page_config(layout="wide")
st.title("COPPER MODELLING AND PREDICTION")
# Creating option menu with purple colored options using Markdown with HTML
selected = option_menu(
    menu_title="",  # Leave it empty since we've already displayed the title above
    options=[
        "About",
        "Selling_Price_Prediction", 
        "Status_Prediction"
    ],
    orientation="horizontal"
)
if selected== 'About':
    st.header('Copper Modelling and Prediction Application')
    st.subheader('Overview')
    st.markdown('Welcome to the Copper Modelling and Prediction application, designed to provide accurate predictions for the selling price and status (win/lose) of copper products based on various input parameters. This tool leverages advanced machine learning techniques to assist stakeholders in making informed decisions.')
    st.markdown(' This application provides a user-friendly interface for predicting the selling price of copper products based on various input parameters. Additionally, it will include functionality to predict the status of copper products in the future. Stay tuned for more updates!')
if selected == 'Selling_Price_Prediction':
    quantity_tons = st.number_input("Quantity (tons)", min_value=0.0, max_value=151.44676637483508)
    customer = st.number_input("Customer", min_value=12458, max_value=30408185)
    country = st.selectbox("Country",["28.","25.","30.","32.","38.","78.","27.","77.","113.","79.","26.","39.","40.","84.","80.","107.","89."])
    status = st.selectbox("Status", ["7","6","5","4","3","2","1","0"])
    item_type = st.selectbox("Item Type",["0","1","2","3","4","5","6"])
    application = st.selectbox("Application",["10.","41.","28.","59.","15.","4.","38.","56.","42.","26.","27.","19.","20.",
       "66.","29.","22.","40.","25.","67.","79.","3.","99.","2.","5.","39.","69.","70.","65.","58.","68."])
    thickness = st.select_slider("Thickness (mm)", options=[f"{value:.1f}" for value in np.arange(0.0, 6.6, 0.1)])

    width = st.select_slider("Width (mm)", options=np.arange(0.0, 1981.0, 1.0))

    product_ref = st.selectbox("Product Reference",["1670798778","1668701718","628377","640665","611993","1668701376","164141591","1671863738","1332077137","640405",
    "1693867550","1665572374","1282007633","1668701698","628117","1690738206","628112","640400","1671876026","164336407",
    "164337175","1668701725","1665572032","611728","1721130331","1693867563","611733","1690738219","1722207579","929423819",
    "1665584320","1665584662","1665584642"])
    df=pd.read_csv('C:/Users/RAMAN/Desktop/DS/PROJECT/Copper_project/Clean_Selling_price.csv')
    del df['Unnamed: 0']
    print(df)
    # Convert categorical variables to numeric
    le_country = LabelEncoder()
    le_status = LabelEncoder()
    le_item_type = LabelEncoder()
    le_application = LabelEncoder()
    le_product_ref = LabelEncoder()
    
    df['country'] = le_country.fit_transform(df['country'])
    df['status'] = le_status.fit_transform(df['status'])
    df['item type'] = le_item_type.fit_transform(df['item type'])
    df['application'] = le_application.fit_transform(df['application'])
    df['product_ref'] = le_product_ref.fit_transform(df['product_ref'])
    def load_model():
        #Split
        from sklearn.model_selection import train_test_split
        x=df.loc[:,df.columns!='selling_price']
        y=df['selling_price']
        # Ensure all feature columns are numeric and handle any conversion issues
        x = x.apply(pd.to_numeric, errors='coerce')

        # Replace infinite values with NaN
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)
        # Fill NaN values with the mean of each column in both training and test sets
        x_train.fillna(x_train.mean(), inplace=True)
        x_test.fillna(x_test.mean(), inplace=True)

        # Check for infinity values in training and test sets
        train_has_inf = x_train.applymap(np.isinf).values.any()
        test_has_inf = x_test.applymap(np.isinf).values.any()

        # Check for NaN values in training and test sets
        train_has_nan = x_train.isna().values.any()
        test_has_nan = x_test.isna().values.any()

        #Scaling
        #Scale(Scaling is not mandatory)
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()
        scaler.fit(x_train)
        x_trained_scaled=scaler.transform(x_train)
        x_test_scaled=scaler.transform(x_test)
        #Model_Fitting
        model = XGBRegressor(learning_rate = 0.6)
        model.fit(x_trained_scaled, y_train)
        return model


    if st.button("Predict Selling Price"):
        # Load the model
       
        model = load_model()

        # Prepare the input data for prediction
        input_data = np.array([[quantity_tons, customer, country, status, item_type, application, thickness, width, product_ref]])
        print(input_data)
        # Convert input data to a NumPy array and ensure all values are floats
        input_data = np.array(input_data, dtype=float)
        print(input_data)
        # Making predictions
        prediction = model.predict(input_data)
        print(prediction)
        # Display the prediction
        st.subheader(f"Predicted Selling Price: ${prediction[0]:,.2f}")
    
if selected == 'Status_Prediction':
    quantity_tons = st.number_input("Quantity (tons)", min_value=0.0, max_value=151.44676637483508)
    customer = st.number_input("Customer", min_value=12458, max_value=30408185)
    country = st.selectbox("Country",["28.","25.","30.","32.","38.","78.","27.","77.","113.","79.","26.","39.","40.","84.","80.","107.","89."])
    item_type = st.selectbox("Item Type",["0","1","2","3","4","5","6"])
    application = st.selectbox("Application",["10.","41.","28.","59.","15.","4.","38.","56.","42.","26.","27.","19.","20.",
       "66.","29.","22.","40.","25.","67.","79.","3.","99.","2.","5.","39.","69.","70.","65.","58.","68."])
    thickness = st.select_slider("Thickness (mm)", options=[f"{value:.1f}" for value in np.arange(0.0, 6.6, 0.1)])

    width = st.select_slider("Width (mm)", options=np.arange(0.0, 1981.0, 1.0))

    product_ref = st.selectbox("Product Reference",["1670798778","1668701718","628377","640665","611993","1668701376","164141591","1671863738","1332077137","640405",
    "1693867550","1665572374","1282007633","1668701698","628117","1690738206","628112","640400","1671876026","164336407",
    "164337175","1668701725","1665572032","611728","1721130331","1693867563","611733","1690738219","1722207579","929423819",
    "1665584320","1665584662","1665584642"])
    selling_price=st.number_input("Selling_price [Min range:243 and Max range:1379]",min_value=251.0,max_value=1371.0)
    df1=pd.read_csv('C:/Users/RAMAN/Desktop/DS/PROJECT/Copper_project/Cleaned_Status.csv')
    del df1['Unnamed: 0']
    print(df1)
    def load_status_model():
        #Split
        from sklearn.model_selection import train_test_split
        x=df1.loc[:,df1.columns!='status']
        y=df1['status']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)
        #Scale
        #Split
        from sklearn.model_selection import train_test_split
        x=df1.loc[:,df1.columns!='status']
        y=df1['status']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)
        #Model
        #Balancing using ENN(Under_sampling and Over_sampling)
        from imblearn.combine import SMOTEENN
        smoteenn=SMOTEENN(random_state=7)
        x_train_smotenn,y_train_smotenn=smoteenn.fit_resample(x_train,y_train)

        #Model
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier()
        model.fit(x_train_smotenn,y_train_smotenn)
        return model
     


    if st.button("Predict Status"):
        # Load the model
        model = load_status_model() 
        # Prepare input data for prediction
        input_data = np.array([[quantity_tons, customer, country, item_type, application, thickness, width, product_ref,selling_price]])
        # Convert input data to a NumPy array and ensure all values are floats
        input_data = np.array(input_data, dtype=float)
        print(input_data)
        # Make prediction
        prediction = model.predict(input_data)
        print(prediction)
        # Display prediction
        if prediction[0] == 1:
            st.success("Prediction: Win")
        else:
            st.error("Prediction: Lose")
