import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))


st.title("Employee Attrition Prediction")
columns = ['Age','Gender', 'Years at Company', 'Monthly Income', 'Job Role', 'Job Satisfaction', 'Number of Promotions', 'Distance from Home','Remote Work','Leadership Opportunities']


age = st.number_input("Enter the age",min_value=15, max_value=50, placeholder="15 - 50")
gender = st.selectbox("Gender",['Male','Female'])
years_at_company = st.number_input("Years at Company",min_value=1, max_value=50, placeholder = "1 - 50")
monthly_income = st.number_input("Monthly Income", min_value = 1000, max_value=16000, placeholder = "1000 - 16000")
job_role = st.selectbox('Job Role',['Education', 'Media', 'Healthcare', 'Technology', 'Finance'])
job_satisfaction = st.selectbox('Job satisfaction', ['Medium', 'High', 'Very High', 'Low'])
number_of_promotions = st.number_input("Enter no. of promotions", min_value=0, max_value=4, placeholder = "0 - 4")
distance_from_home = st.number_input("Distance from home in miles",min_value = 1, max_value = 100, placeholder = "1 - 100")
remote_work = st.selectbox("works Remotely",['No', 'Yes'])
leadership_opportunities = st.selectbox("got leadership opportunities", ['No', 'Yes'])

if st.button("predict"):
    input_data = pd.DataFrame([[age, gender, years_at_company, monthly_income, job_role, job_satisfaction, number_of_promotions, distance_from_home, remote_work, leadership_opportunities]], columns=columns)
    input_data_transformed = preprocessor.transform(input_data)
    pred = model.predict(input_data_transformed)

    if pred == 1:
        st.title("Employee will stay")
    elif pred == 0:
        st.title("Employee will leave")
