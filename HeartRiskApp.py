import streamlit as st
import pandas as pd
import numpy as np
import joblib

# to run 
# streamlit run HeartRiskApp.py

st.title("Heart Disease Risk Prediction App")

st.sidebar.title('Navigation')
selection = st.sidebar.radio('',['Heart Disease Risk Prediction App','Project Description','Author'])

if selection == 'Heart Disease Risk Prediction App':
    model = joblib.load('rf_model.pkl')

    st.markdown(
        """
        Use this interactive tool to predict your risk of heart disease based on clinical parameters.  
        """
    )
    st.subheader("📝 **Provide Your Details**")

    # Load dataset to extract unique categorical options
    url = "https://github.com/AyushmanGHub/Heartbeat-Sentinel_Decoding-and-Predicting-Heart-Failure/raw/main/HeartDataset.csv"
    data = pd.read_csv(url)

      # Create three columns for better layout and spacing
    col1, spacer, col2 = st.columns([1.4, 0.2, 1.4])  # Adjust the width ratios for desired gap size

    with col1:
        st.markdown("#### General Information")

        st.markdown("<br>", unsafe_allow_html=True)
        st.slider("📅 Age (in years)", min_value=20, max_value=80, value=50, step=1, key="age")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.selectbox("⚤ Sex", options=['Male (M)', 'Female (F)'], key="sex")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.selectbox(
                "💓 Chest Pain Type (CPT)",
                options=['ATA (Atypical Angina)', 'NAP (Non-Anginal Pain)', 'ASY (Asymptomatic)', 'TA (Typical Angina)'],
                key="cp"
            )
    
        st.markdown("<br>", unsafe_allow_html=True)
        st.slider("🩺 Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1, key="resting_bp")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.slider("🍔 Serum Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1, key="cholesterol")
        

    with col2:
        st.markdown("#### Diagnostic Parameters")

        st.markdown("<br>", unsafe_allow_html=True)
        st.selectbox("🍬 Fasting Blood Sugar", options=["No (0)", "Yes (1)"], key="fasting_bs")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.selectbox(
                "📉 Resting Electrocardiographic Results",
                options=['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
                key="resting_ecg"
            )

        st.markdown("<br>", unsafe_allow_html=True)     
        st.slider("🏃‍♂️ Maximum Heart Rate Achieved (MaxHR)", min_value=60, max_value=220, value=150, step=1, key="max_hr")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.selectbox("🛑 Exercise Induced Angina", options=["No (N)", "Yes (Y)"], key="exercise_angina")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.slider("📉 Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, key="oldpeak")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.selectbox(
                "📈  ST Segment Slope",
                options=['Unsloping (Up)', 'Flat', 'Downsloping (Down)'],
                key="st_slope"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)

    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [st.session_state["age"]],
        'Sex': [1 if 'Male' in st.session_state["sex"] else 0],
        'ChestPainType': [st.session_state["cp"].split(' ')[0]],
        'RestingBP': [st.session_state["resting_bp"]],
        'Cholesterol': [st.session_state["cholesterol"]],
        'FastingBS': [int(st.session_state["fasting_bs"].split('(')[1][0])],
        'RestingECG': [st.session_state["resting_ecg"].split(' ')[0]],
        'MaxHR': [st.session_state["max_hr"]],
        'ExerciseAngina': [1 if 'Yes' in st.session_state["exercise_angina"] else 0],
        'Oldpeak': [st.session_state["oldpeak"]],
        'ST_Slope': [st.session_state["st_slope"].split(' ')[0]]
    })

    # Encode categorical features (same as during model training)
    input_data['ChestPainType'] = input_data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    input_data['RestingECG'] = input_data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    input_data['ST_Slope'] = input_data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

    # Make prediction
    st.subheader("🔍 **Prediction Results**")

    if st.button("🔎 Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ **The model predicts a HIGH risk of heart disease!**")
        else:
            st.success("✅ **The model predicts a LOW risk of heart disease.**")

    st.markdown(
        """
        The prediction is based on the information you provided.
        \\
        **Disclaimer:** This is for informational purposes only and should not be considered medical advice.
        """
    )
    

        

        

elif selection == 'Project Description':
    # Project Title
    st.title("🔍 Heart Disease Prediction Using Machine Learning")

    # Project Overview
    st.header("🩺 **Project Description**")

    st.markdown("""
    Cardiovascular diseases (CVDs), commonly referred to as **heart diseases**, are a leading cause of mortality worldwide. 
    This project addresses the critical need for **early diagnosis and risk prediction** by leveraging historical health data 
    and advanced machine learning techniques to predict heart failure risk.
    """)

    # About the Machine Learning Approach
    st.subheader("🚀 **Machine Learning Approach**")
    st.markdown("""
    This project employs **Random Forest**, a powerful ensemble machine learning algorithm, on a diverse dataset. 
    Key features include:
    - **Age**, **Gender**, **Cholesterol Levels**, **Blood Pressure**, and other clinical parameters.
    - The **target variable** was excluded during preprocessing to avoid data leakage.
    - The model was trained on the remaining features to ensure reliable predictions.
    """)

    # Highlight Random Forest
    st.subheader("🌳 **Why Random Forest?**")
    st.markdown("""
    **Random Forest** is an ensemble learning technique based on multiple **decision trees**, offering high accuracy and robustness:
    - Each tree is trained on a **random subset** of data and features, reducing overfitting.
    - It aggregates predictions across all trees to deliver **reliable results**.
    - Handles **non-linear relationships** effectively, making it suitable for complex datasets like ours.
    """)
    st.info("In this project, Random Forest was trained on parameters like **Age**, **Cholesterol Levels**, and **Chest Pain Type** to predict heart disease risk. The model demonstrated high **accuracy** and **interpretability**, ensuring reliable predictions.")

    # Model Performance
    st.subheader("📊 **Model Performance**")
    st.markdown("""
    The models' performance was evaluated using the following metrics:
    - **Accuracy**: Measures the overall correctness of predictions.
    - **Precision**: Indicates the proportion of true positive predictions.
    - **Recall**: Reflects the model’s ability to detect true positives.
    - **F1 Score**: Balances precision and recall into a single metric.
    - **ROC-AUC Curve**: Highlights the model's ability to distinguish between classes.
    """)
    st.success("The models provided **robust predictions**, offering valuable insights for medical professionals to facilitate early intervention and personalized treatment strategies.")

    # Resources Section
    st.subheader("📂 **Resources**")
    st.markdown("""
    Explore the full project and dataset on my **GitHub repository**:
    [🔗 Heartbeat Sentinel: Decoding and Predicting Heart Failure](https://github.com/AyushmanGHub/Heartbeat-Sentinel_Decoding-and-Predicting-Heart-Failure)
    """)

    # Add a Decorative Divider
    st.markdown("---")
  
elif selection == 'Author':
    # Author Section
    st.header("👨‍💻 **About the Author**")

    # # Add a photo (optional, if you have one)
    # st.image("https://via.placeholder.com/150", width=120)  # Replace URL with your profile picture link

    st.markdown("""
    This app was created by **[Ayushman Anupam](https://github.com/AyushmanGHub)**, 
    a passionate data enthusiast with a keen interest in **Healthcare Data Analysis** and **Disease Prediction**.  
    The project demonstrates the use of **machine learning techniques** to address real-world challenges in the medical domain.
    """)

    # GitHub Section
    st.subheader("📂 **Resources**")
    st.markdown("""
    The codebase and all project resources are publicly available on GitHub.  
    🔗 **[Explore the Project Repository](https://github.com/AyushmanGHub/Heartbeat-Sentinel_Decoding-and-Predicting-Heart-Failure)**  
    """)

    # LinkedIn Section
    st.subheader("📬 **Connect with Me**")
    st.markdown("""
    Let's connect! I'm always open to meaningful conversations, collaborations, and sharing insights on data-driven projects.  
    💼 **[LinkedIn: Ayushman Anupam](https://www.linkedin.com/in/ayushman-anupam/)**  
    Feel free to reach out and share your thoughts or ideas!
    """)

    # Add decorative divider
    st.markdown("---")
