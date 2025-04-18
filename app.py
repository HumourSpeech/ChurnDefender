import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# load the model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.set_page_config(page_title="Churn Defender", page_icon=":shield:", layout="centered")
st.title('🔮 Churn Defender')
st.markdown("""
Welcome to **Churn Defender**, a machine learning-powered application to predict customer churn. 
Enter customer details below to find out the likelihood of churn. Let's make data-driven decisions together! 🚀
""")
#User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

#One_hot encode 'Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combine one_hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scaling the input data
input_data_scaled = scaler.transform(input_data)

#prediction churn
# prediction = model.predict(input_data_scaled)
# pred_probability = prediction[0][0]

# st.write(f'Churn Probabilty: {pred_probability: .2f}')

# if pred_probability > 0.5:
#     st.write('Customer is likely to churn')
# else:
#     st.write('Customer is not likely to churn')

# Predict churn
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    pred_prob = prediction[0][0]
    if pred_prob > 0.5:
        st.error(f"⚠️ High Risk of Churn: {pred_prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Churn: {pred_prob:.2f}")

st.markdown("""
---
### Developed by [Nitin Mishra](https://github.com/HumourSpeech/ChurnDefender/blob/main/README.md)
""")