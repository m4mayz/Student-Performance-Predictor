import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import plotly.graph_objects as go
import plotly.express as px
import kagglehub

# Set page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fun messages for each grade
grade_messages = {
    'A': [
    "Wow, you're a total genius! With this score, you could be a lecturer!",
    "Incredible! Your grade makes NASA want to recruit you!",
    "Academic royalty! Your brain is razor-sharp!",
    "Your achievement is on fire! Even the AC can't keep up!",
    "Five-star! The Michelin Guide wants to give you an academic award!",
    "Perfect shot! The academic cupid hit the bullseye!"
    ],
    'B': [
        "Great job! You've reached the hardworking student level!",
        "Solid! Your score is like a wall—strong and sturdy!",
        "Good job! You hit your target nicely!",
        "Colorful! Your achievement shines like a rainbow!",
        "Cruise control! Smooth and steady—awesome!",
        "Apple of my eye! Your teacher must be super proud!"
    ],
    'C': [
        "Not bad! Still room to become an academic superhero!",
        "Like a plant, you're still growing and evolving!",
        "Ups and downs are normal—what matters is moving forward!",
        "Like pizza—still great even if not perfect!",
        "Slow but sure! Even a turtle can win the race!",
        "A sunrise moment! Tomorrow will be brighter!"
    ],
    'D': [
        "Don’t give up! Even Batman fell before he flew!",
        "This is just the sunrise—still a full day of chances ahead!",
        "Under construction! Improvements in progress!",
        "Life’s a circus—sometimes you're the clown, sometimes the star!",
        "Bad luck today? Good luck tomorrow! Keep fighting!",
        "Battery low? Time to recharge and come back stronger!"
    ],
    'E': [
        "No excuses, you're very stupid"
    ]
}

@st.cache_data
def load_and_process_data():
    # Download dataset
    path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
    df = pd.read_csv(path + "/StudentPerformanceFactors.csv")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)

    # Handle values exceeding logical limits
    if 'Attendance' in df.columns:
        df['Attendance'] = df['Attendance'].clip(upper=100)
    
    if 'Exam_Score' in df.columns:
        df['Exam_Score'] = df['Exam_Score'].clip(upper=100, lower=0)
    
    if 'Previous_Scores' in df.columns:
        df['Previous_Scores'] = df['Previous_Scores'].clip(upper=100)
    
    if 'Sleep_Hours' in df.columns:
        df['Sleep_Hours'] = df['Sleep_Hours'].clip(upper=24)
    
    if 'Hours_Studied' in df.columns:
        df['Hours_Studied'] = df['Hours_Studied'].clip(upper=168)
    
    if 'Physical_Activity' in df.columns:
        df['Physical_Activity'] = df['Physical_Activity'].clip(upper=168)

    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

@st.cache_resource
def train_model():
    df = load_and_process_data()
    
    # Define features
    numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                         'Tutoring_Sessions', 'Physical_Activity']

    categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                           'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                           'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                           'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    
    # Encode categorical variables
    df_encoded = df.copy()
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature])
            label_encoders[feature] = le

    # Feature selection using SelectKBest
    X_temp = df_encoded.drop('Exam_Score', axis=1)
    y_temp = df_encoded['Exam_Score']

    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X_temp, y_temp)

    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X_temp.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)

    # Select features with score > 10
    important_features = feature_scores[feature_scores['Score'] > 10]['Feature'].tolist()

    # Create cleaned dataset
    df_cleaned = df_encoded[important_features + ['Exam_Score']].copy()
    
    # Prepare features and target
    X = df_cleaned.drop('Exam_Score', axis=1)
    y = df_cleaned['Exam_Score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Calculate model performance
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return model, scaler, label_encoders, important_features, model_metrics

def assign_grade(score):
    if score >= 80:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 60:
        return 'C'
    elif score >= 50:
        return 'D'
    else:
        return 'E'

def get_grade_color(grade):
    colors = {'A': '#00ff00', 'B': '#ffff00', 'C': '#ff8c00', 'D': '#ff0000'}
    return colors.get(grade, '#gray')

def create_gauge_chart(score, grade, accuracy):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': (
                f"Exam Score<br>"
                f"<span style='font-size:0.8em;color:{get_grade_color(grade)}'>Grade: {grade}</span><br>"
            )
        },
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_grade_color(grade)},
            'steps': [
                {'range': [0, 55], 'color': "lightgray"},
                {'range': [55, 75], 'color': "lightyellow"},
                {'range': [75, 85], 'color': "lightgreen"},
                {'range': [85, 100], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    # Title and description
    st.title("🎓 Student Performance Predictor")
    st.markdown("### Exam score prediction based on various academic and personal factors.")
    st.markdown("*Based on dataset [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data)*")
    
    # Load model
    try:
        with st.spinner("Loading models and data..."):
            model, scaler, label_encoders, important_features, model_metrics = train_model()
            df = load_and_process_data()  # Load data for min/max

        # Tampilkan model accuracy di bawah judul utama
        accuracy = model_metrics['R2'] * 100
        st.info(f"**Model Accuracy: {accuracy:.1f}%**")

        # Display model performance in expander
        with st.expander("📊 Model Performance Metrics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{model_metrics['R2']:.4f}")
            with col2:
                st.metric("RMSE", f"{model_metrics['RMSE']:.4f}")
            with col3:
                st.metric("MAE", f"{model_metrics['MAE']:.4f}")
            with col4:
                st.metric("MSE", f"{model_metrics['MSE']:.4f}")
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Sidebar for inputs
    st.sidebar.header("📝 Input Student Features")
    st.sidebar.markdown("Enter student information for prediction (only using features with score > 10):")
    
    # Create input fields based on important features
    input_values = {}
    
    # Numerical inputs
    st.sidebar.subheader("📊 Numerical Data")
    
    # Ambil min, max, dan median dari dataset untuk setiap fitur numerik
    numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                         'Tutoring_Sessions', 'Physical_Activity']
    numerical_labels = {
        'Hours_Studied': "Study Hours per Week",
        'Attendance': "Attendance Rate (%)",
        'Sleep_Hours': "Sleep Hours per Day",
        'Previous_Scores': "Previous Scores",
        'Tutoring_Sessions': "Tutoring Sessions per Month",
        'Physical_Activity': "Physical Activity (hours/week)"
    }
    for feature in important_features:
        if feature in numerical_features and feature in df.columns:
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            median_val = int(df[feature].median())
            label = numerical_labels.get(feature, feature)
            input_values[feature] = st.sidebar.slider(label, min_val, max_val, median_val)
    
    # Categorical inputs
    st.sidebar.subheader("📋 Categorical Data")
    
    # Define categorical options
    categorical_options = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Access_to_Resources': ['Low', 'Medium', 'High'],
        'Extracurricular_Activities': ['Yes', 'No'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Internet_Access': ['Yes', 'No'],
        'Family_Income': ['Low', 'Medium', 'High'],
        'Teacher_Quality': ['Low', 'Medium', 'High'],
        'School_Type': ['Public', 'Private'],
        'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
        'Learning_Disabilities': ['Yes', 'No'],
        'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
        'Distance_from_Home': ['Near', 'Moderate', 'Far'],
        'Gender': ['Male', 'Female']
    }
    
    # Create input fields for categorical features that are in important_features
    for feature in important_features:
        if feature in categorical_options:
            options = categorical_options[feature]
            # Create readable labels
            label = feature.replace('_', ' ').title()
            if feature == 'Parental_Involvement':
                label = "Parental Involvement"
            elif feature == 'Access_to_Resources':
                label = "Access to Learning Resources"
            elif feature == 'Extracurricular_Activities':
                label = "Extracurricular Activities"
            elif feature == 'Motivation_Level':
                label = "Motivation Level"
            elif feature == 'Internet_Access':
                label = "Internet Access"
            elif feature == 'Family_Income':
                label = "Family Income"
            elif feature == 'Teacher_Quality':
                label = "Teacher Quality"
            elif feature == 'School_Type':
                label = "School Type"
            elif feature == 'Peer_Influence':
                label = "Peer Influence"
            elif feature == 'Learning_Disabilities':
                label = "Learning Disabilities"
            elif feature == 'Parental_Education_Level':
                label = "Parental Education Level"
            elif feature == 'Distance_from_Home':
                label = "Distance from Home"
            elif feature == 'Gender':
                label = "Gender"
            
            default_idx = len(options) // 2 if len(options) > 2 else 0
            input_values[feature] = st.sidebar.selectbox(label, options, index=default_idx)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Result Prediction")
        
        if st.button("🔮 Predict Now!", type="primary", use_container_width=True):
            # Prepare input data
            input_df = pd.DataFrame([input_values])
            
            # Encode categorical variables
            for feature in important_features:
                if feature in label_encoders:
                    try:
                        input_df[feature] = label_encoders[feature].transform(input_df[feature])
                    except ValueError:
                        # Handle unseen categories by using the most frequent class
                        input_df[feature] = 0
            
            # Ensure all important features are present
            for feature in important_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[important_features]
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            predicted_score = model.predict(input_scaled)[0]
            predicted_score = max(0, min(100, predicted_score))  # Clip to valid range
            grade = assign_grade(predicted_score)
            
            # Display results
            st.success(f"**Exam Score Prediction: {predicted_score:.1f}**")
            st.info(f"**Performance Grade: {grade}**")
            
            # Display fun message
            fun_message = random.choice(grade_messages[grade])
            st.markdown(f"### 🎉 {fun_message}")
            
            # Store results in session state for visualization
            st.session_state.predicted_score = predicted_score
            st.session_state.grade = grade

    with col2:
        st.subheader("📈 Result Visualization")
        
        if hasattr(st.session_state, 'predicted_score'):
            # Hitung akurasi model (R2 * 100)
            accuracy = model_metrics['R2'] * 100
            # Gauge chart
            gauge_fig = create_gauge_chart(st.session_state.predicted_score, st.session_state.grade, accuracy)
            st.plotly_chart(gauge_fig, use_container_width=True)
        else:
            st.info("Click the 'Predict Now!' button to see the visualization of the results")
    
    # Additional info
    st.markdown("---")
    st.subheader("📚 Assessment System Information")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**Grade A** 🌟")
        st.markdown("80-100 poin")
        st.markdown("Excellent")
    
    with col2:
        st.markdown("**Grade B** 👍")
        st.markdown("70-79 poin")
        st.markdown("Good")
    
    with col3:
        st.markdown("**Grade C** 😊")
        st.markdown("60-69 poin")
        st.markdown("Average")
    
    with col4:
        st.markdown("**Grade D** 💪")
        st.markdown("50-59 poin")
        st.markdown("Needs Improvement")
        
    with col5:
        st.markdown("**Grade E** ❌")
        st.markdown("0-49 poin")
        st.markdown("Poor Performance")
    
    # Display important features
    st.markdown("---")
    st.subheader("🔍 Feature Importance (F-regression)")

    # Ambil ulang skor feature importance dari SelectKBest
    df_encoded = df.copy()
    categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                           'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                           'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                           'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    for feature in categorical_features:
        if feature in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature])

    X_temp = df_encoded.drop('Exam_Score', axis=1)
    y_temp = df_encoded['Exam_Score']

    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X_temp, y_temp)

    feature_scores = pd.DataFrame({
        'Feature': X_temp.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=True)  # ascending for horizontal bar

    # Plot bar chart
    fig = px.bar(
        feature_scores,
        x='Score',
        y='Feature',
        orientation='h',
        title="Feature Importance Scores (F-regression)",
        labels={'Score': 'F-score', 'Feature': 'Feature'},
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()