import streamlit as st
import pandas as pd

from dotenv import load_dotenv

from api.prediction import get_pred
 
load_dotenv(dotenv_path=".dev.env")

st.title("Iris Prediction app")
st.text("Lorem ipsum dolor, sit amet consectetur adipisicing elit. Temporibus minus neque, voluptas, unde tempore totam qui laboriosam mollitia pariatur quis labore. Possimus facilis itaque commodi accusamus earum, et tenetur exercitationem.")
st.divider()

with st.form("predict_form"):
    sepal_length = st.number_input(label="Sepal Length", min_value=0)
    sepal_width = st.number_input(label="Sepal Width", min_value=0)
    petal_length = st.number_input(label="Petal Length", min_value=0)
    petal_width = st.number_input(label="Petal Width", min_value=0)

    print(f"Sepal Length: {sepal_length}, Sepal Width: {sepal_width}, Petal Length: {petal_length}, Petal Width: {petal_width}")
    submit = st.form_submit_button(label="Submit")

    data = {
        "sepal_length":sepal_length,
        "sepal_width":sepal_width,
        "petal_length":petal_length,
        "petal_width":petal_width,
    }
    result = get_pred(data=data)    
    st.write(f"{result[0]} With result {result[1]}")
