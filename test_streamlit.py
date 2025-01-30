import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Load the trained model directly without a scaler
model_path = '/Users/sobithav/Documents/ml_project/trained_model.sav'  # Path to your trained model
loaded_model = pickle.load(open(model_path, 'rb'))

def get_clean_data():
    # Assuming the input data requires some preprocessing
    data = pd.read_csv("/Users/sobithav/Dropbox/diabetic/diabetes.csv")  # Adjust this path as needed
    data = data.drop(["Unnamed: 32", 'id'], axis=1)
    data['diagnosis'] = data["diagnosis"].map({'M': 1, "B": 0})
    return data

def add_sidebar():
    st.sidebar.header("Diabetes Prediction")

    slider_labels = [
        ("Pregnancies", "pregnancies"),
        ("Glucose", "glucose"),
        ("Blood Pressure", "blood_pressure"),
        ("Skin Thickness", "skin_thickness"),
        ("Insulin", "insulin"),
        ("BMI", "bmi"),
        ("Diabetes Pedigree Function", "diabetes_pedigree_function"),
        ("Age", "age"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0,
            max_value=200,  # Adjust this range based on your dataset
            value=0,  # Default value
            step=1
        )

    return input_dict

def add_predictions(input_data):
    # Use the already loaded model without scaling
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Prediction using the loaded model
    prediction = loaded_model.predict(input_array)

    st.subheader("Cell Cluster Prediction")
    st.markdown(f"<h4 class='prediction-header'>The cell cluster is:</h4>", unsafe_allow_html=True)

    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)
    
    # Probability calculation if available
    if hasattr(loaded_model, 'predict_proba'):
        prob_benign = loaded_model.predict_proba(input_array)[0][0]
        prob_malignant = loaded_model.predict_proba(input_array)[0][1]
        st.write(f"**The Probability of being benign is:** {prob_benign:.2f}")
        st.write(f"**The Probability of being malignant is:** {prob_malignant:.2f}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def get_radar_chart(input_data):
    # Directly using the raw input data (no scaling)
    input_data_scaled = input_data  # No scaling applied
    categories = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI',
                  'Diabetes Pedigree Function', 'Age']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(input_data_scaled.values()),  # Values for radar chart
        theta=categories,
        fill='toself',
        name='Input Data'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 200]  # Adjust the range as per the input values' expected range
            )),
        showlegend=True
    )
    return fig

def main():
    st.set_page_config(
        page_title="Diabetes Prediction",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for styling (using inline styles as you don't have a separate CSS file)
    st.markdown("""
    <style>
    .diagnosis {
        font-size: 24px;
        font-weight: bold;
    }
    .benign {
        color: green;
        text-transform: uppercase;
    }
    .malignant {
        color: red;
        text-transform: uppercase;
    }
    .prediction-header {
        font-size: 20px;
        color: #333;
        font-weight: bold;
    }
    .sidebar .stSlider {
        margin-bottom: 15px;
    }
    hr {
        border: 0;
        border-top: 1px solid #ccc;
        margin: 20px 0;
    }
    .stButton > button {
        background-color: #00BFFF;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #009ACD;
    }
    .stText {
        font-size: 16px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Diabetes Prediction App")
        st.write("This app predicts whether a person is diabetic based on certain health parameters. "
                 "Use the sidebar to enter values and predict the result.")

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
