import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier    
import numpy as np
iris= load_iris()
st.title("Iris Flower Classification App")
if st.button("Show Dataset"):
    st.write("Iris Dataset Sample (first 10 rows)")
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    st.write(df_iris.head(10))

st.sidebar.header("User Input Features")
def user_input_features():
    sepal_length= st.sidebar.slider("Sepal Length", 4.3,7.9,5.4)
    sepal_width= st.sidebar.slider("Sepal Width", 2.0,4.4,3.4)
    petal_length= st.sidebar.slider("Petal Length", 1.0,6.9,1.3)
    petal_width= st.sidebar.slider("Petal Width", 0.1,2.5,0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features= pd.DataFrame(data, index=[0])
    return features
df= user_input_features()
st.subheader("User Input Features")
st.write(df)
model = RandomForestClassifier()
model.fit(iris.data, iris.target)
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.subheader("Prediction")
st.write(iris.target_names[prediction][0])
st.subheader("Prediction Probability")
st.write(prediction_proba)
st.bar_chart(prediction_proba, use_container_width=True)


st.markdown("---")
st.caption("Built with ❤️ using Streamlit by Deepti")
