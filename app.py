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
        "🌟 Wah, kamu jenius banget! Nilai segini mah bisa jadi dosen! 😎",
        "🚀 Mantap jiwa! Nilaimu bikin NASA pengen rekrut kamu! 🛸",
        "👑 Raja/Ratu akademik nih! Otak encer banget! 💎",
        "🔥 Hot banget prestasinya! AC ruangan sampai kewalahan! ❄️",
        "⭐ Bintang lima! Michelin Guide mau kasih award akademik nih! 🏆",
        "🎯 Perfect shot! Panah cupid akademik mengenai sasaran! 💘"
    ],
    'B': [
        "👍 Bagus banget! Kamu udah level mahasiswa yang rajin nih! 📚",
        "💪 Solid! Nilaimu seperti tembok, kuat dan kokoh! 🧱",
        "🎯 Good job! Target tercapai dengan apik! 🏹",
        "🌈 Colorful! Prestasimu cerah seperti pelangi! 🎨",
        "🚗 Cruise control! Stabil dan terkendali, mantap! 🛣️",
        "🍎 Apple of my eye! Guru pasti bangga banget! 🍏"
    ],
    'C': [
        "😊 Lumayan nih! Masih ada ruang untuk jadi superhero akademik! 🦸‍♂️",
        "🌱 Seperti tanaman, masih terus tumbuh dan berkembang! 🌿",
        "🎢 Naik turun itu wajar, yang penting terus maju! 🚂",
        "🍕 Seperti pizza, masih enak meski belum perfect! 🧀",
        "🐢 Slow but sure! Kura-kura juga bisa menang lho! 🏁",
        "🌅 Sunrise moment! Besok pasti lebih cerah! ☀️"
    ],
    'D': [
        "💪 Jangan menyerah! Even Batman pernah jatuh sebelum terbang! 🦇",
        "🌅 Ini baru sunrise, masih banyak kesempatan hari ini! ☀️",
        "🚧 Under construction! Sedang dalam tahap perbaikan! 🔨",
        "🎪 Life is a circus, sometimes you're the clown, sometimes the star! 🤹‍♂️",
        "🍀 Bad luck today? Good luck tomorrow! Keep fighting! 💪",
        "🔋 Battery low? Time to recharge and comeback stronger! ⚡"
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
    
    # Define features as in original code
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

    # Select top features (top 60% of features)
    n_features_to_keep = int(len(feature_scores) * 0.6)
    important_features = feature_scores.head(n_features_to_keep)['Feature'].tolist()

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
    if score >= 85:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 55:
        return 'C'
    else:
        return 'D'

def get_grade_color(grade):
    colors = {'A': '#00ff00', 'B': '#ffff00', 'C': '#ff8c00', 'D': '#ff0000'}
    return colors.get(grade, '#gray')

def create_gauge_chart(score, grade):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Exam Score<br><span style='font-size:0.8em;color:{get_grade_color(grade)}'>Grade: {grade}</span>"},
        delta = {'reference': 75},
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

def create_probability_chart(score):
    # Simulate confidence interval around prediction
    std_dev = 5  # Standard deviation for prediction uncertainty
    x = np.linspace(max(0, score-20), min(100, score+20), 100)
    y = np.exp(-0.5 * ((x - score) / std_dev) ** 2)
    y = y / np.sum(y) * 100  # Convert to percentage
    
    fig = px.area(x=x, y=y, 
                  title="Prediction Confidence Distribution",
                  labels={'x': 'Possible Exam Scores', 'y': 'Probability (%)'},
                  color_discrete_sequence=['lightblue'])
    
    fig.add_vline(x=score, line_dash="dash", line_color="red", 
                  annotation_text=f"Predicted: {score:.1f}")
    
    # Add confidence intervals
    fig.add_vrect(x0=score-std_dev, x1=score+std_dev, 
                  fillcolor="yellow", opacity=0.2,
                  annotation_text="68% Confidence", annotation_position="top left")
    
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
    st.sidebar.markdown("Enter student information for prediction:")
    
    # Create input fields based on important features
    input_values = {}
    
    # Numerical inputs
    st.sidebar.subheader("📊 Numerical Data")
    
    # Define default values and ranges for numerical features
    numerical_defaults = {
        'Hours_Studied': (15, 0, 50, "Study Hours per Week"),
        'Attendance': (85, 0, 100, "Attendance Rate (%)"),
        'Sleep_Hours': (7, 3, 12, "Sleep Hours per Day"),
        'Previous_Scores': (75, 0, 100, "Previous Scores"),
        'Tutoring_Sessions': (2, 0, 10, "Tutoring Sessions per Month"),
        'Physical_Activity': (5, 0, 20, "Physical Activity (hours/week)")
    }
    
    for feature in important_features:
        if feature in numerical_defaults:
            default, min_val, max_val, label = numerical_defaults[feature]
            input_values[feature] = st.sidebar.slider(label, min_val, max_val, default)
    
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
            
            # Calculate prediction probabilities for each grade
            grade_probs = {
                'A': max(0, min(100, 100 * (1 / (1 + np.exp(-0.1 * (predicted_score - 85)))))),
                'B': max(0, min(100, 100 * (1 / (1 + np.exp(-0.2 * (predicted_score - 75)))) - 
                         100 * (1 / (1 + np.exp(-0.1 * (predicted_score - 85)))))),
                'C': max(0, min(100, 100 * (1 / (1 + np.exp(-0.2 * (predicted_score - 55)))) - 
                         100 * (1 / (1 + np.exp(-0.2 * (predicted_score - 75)))))),
                'D': max(0, min(100, 100 - 100 * (1 / (1 + np.exp(-0.2 * (predicted_score - 55))))))
            }
            
            # Normalize probabilities
            total_prob = sum(grade_probs.values())
            if total_prob > 0:
                grade_probs = {k: v/total_prob*100 for k, v in grade_probs.items()}
            
            # Display prediction probabilities
            st.subheader("📊 Prediction Probabilities")
            for grade_key, prob in grade_probs.items():
                st.write(f"**Grade {grade_key}:** {prob:.1f}%")
            
            # Store results in session state for visualization
            st.session_state.predicted_score = predicted_score
            st.session_state.grade = grade
            st.session_state.grade_probs = grade_probs
    
    with col2:
        st.subheader("📈 Result Visualization")
        
        if hasattr(st.session_state, 'predicted_score'):
            # Gauge chart
            gauge_fig = create_gauge_chart(st.session_state.predicted_score, st.session_state.grade)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Probability chart
            prob_fig = create_probability_chart(st.session_state.predicted_score)
            st.plotly_chart(prob_fig, use_container_width=True)
            
            # Grade probabilities bar chart
            if hasattr(st.session_state, 'grade_probs'):
                grades = list(st.session_state.grade_probs.keys())
                probs = list(st.session_state.grade_probs.values())
                colors = [get_grade_color(g) for g in grades]
                
                fig_bar = go.Figure(data=[go.Bar(x=grades, y=probs, marker_color=colors)])
                fig_bar.update_layout(
                    title="Grade Prediction Probabilities",
                    xaxis_title="Grade",
                    yaxis_title="Probability (%)",
                    height=300
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Click the 'Predict Now!' button to see the visualization of the results")
    
    # Additional info
    st.markdown("---")
    st.subheader("📚 Assessment System Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Grade A** 🌟")
        st.markdown("85-100 poin")
        st.markdown("Excellent")
    
    with col2:
        st.markdown("**Grade B** 👍")
        st.markdown("75-84 poin")
        st.markdown("Good")
    
    with col3:
        st.markdown("**Grade C** 😊")
        st.markdown("55-74 poin")
        st.markdown("Average")
    
    with col4:
        st.markdown("**Grade D** 💪")
        st.markdown("0-54 poin")
        st.markdown("Needs Improvement")
    
    # Display important features
    st.markdown("---")
    st.subheader("🔍 Important Features for Prediction")
    st.write("The following features were selected based on Feature Selection (SelectKBest):")
    
    # Create columns for features
    feature_cols = st.columns(3)
    for i, feature in enumerate(important_features):
        with feature_cols[i % 3]:
            st.write(f"• {feature.replace('_', ' ').title()}")

if __name__ == "__main__":
    main()