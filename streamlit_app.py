#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import streamlit as st 
from sqlalchemy import create_engine
import joblib, pickle


# In[10]:


import os
os.chdir(r"G:\iNeuron.ai\Project")


# In[4]:


# Load model and preprocessing objects
model = pickle.load(open('svc.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winzor')
scale = joblib.load('standard')
pca = joblib.load('pca')

# In[12]:


def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    clean = pd.DataFrame(impute.transform(data),columns=data.columns)
    clean1 = pd.DataFrame(winzor.transform(clean),columns=clean.columns)
    clean2 = pd.DataFrame(scale.transform(clean1),columns=clean1.columns)
    clean3 = pd.DataFrame(pca.transform(clean2))
    prediction = pd.DataFrame(model.predict(clean3), columns=['Prediction'])
    prediction['Prediction'] = prediction['Prediction'].map({0: 'No Default', 1: 'Default'})
    
    final = pd.concat([data, prediction], axis=1)
    final.to_sql('credit_card_default', con=engine, if_exists='replace', index=False)
    
    return final



# In[13]:


def main():  

    st.sidebar.title("Credit Card Default Prediction")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Default Prediction</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "")
    pw = st.sidebar.text_input("password", "", type="password")
    db = st.sidebar.text_input("database", "")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
                                   
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm))


# In[15]:


if __name__=='__main__':
    main()


# In[ ]:




