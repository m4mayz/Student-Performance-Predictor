import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

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
        "⭐ Bintang lima! Michelin Guide mau kasih award akademik nih! 🏆"
    ],
    'B': [
        "👍 Bagus banget! Kamu udah level mahasiswa yang rajin nih! 📚",
        "💪 Solid! Nilaimu seperti tembok, kuat dan kokoh! 🧱",
        "🎯 Good job! Target tercapai dengan apik! 🏹",
        "🌈 Colorful! Prestasimu cerah seperti pelangi! 🎨",
        "🚗 Cruise control! Stabil dan terkendali, mantap! 🛣️"
    ],
    'C': [
        "😊 Lumayan nih! Masih ada ruang untuk jadi superhero akademik! 🦸‍♂️",
        "🌱 Seperti tanaman, masih terus tumbuh dan berkembang! 🌿",
        "🎢 Naik turun itu wajar, yang penting terus maju! 🚂",
        "🍕 Seperti pizza, masih enak meski belum perfect! 🧀",
        "🐢 Slow but sure! Kura-kura juga bisa menang lho! 🏁"
    ],
    'D': [
        "💪 Jangan menyerah! Even Batman pernah jatuh sebelum terbang! 🦇",
        "🌅 Ini baru sunrise, masih banyak kesempatan hari ini! ☀️",
        "🚧 Under construction! Sedang dalam tahap perbaikan! 🔨",
        "🎪 Life is a circus, sometimes you're the clown, sometimes the star! 🤹‍♂️",
        "🍀 Bad luck today? Good luck tomorrow! Keep fighting! 💪"
    ]
}

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset for training"""
    # This is a placeholder - in real deployment, you'd load actual data
    # For demo purposes, creating synthetic data based on the features
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Hours_Studied': np.random.normal(15, 5, n_samples).clip(0, 50),
        'Attendance': np.random.normal(80, 15, n_samples).clip(0, 100),
        'Sleep_Hours': np.random.normal(7, 1.5, n_samples).clip(3, 12),
        'Previous_Scores': np.random.normal(75, 15, n_samples).clip(0, 100),
        'Tutoring_Sessions': np.random.poisson(2, n_samples).clip(0, 10),
        'Physical_Activity': np.random.normal(5, 3, n_samples).clip(0, 20),
        'Parental_Involvement': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Access_to_Resources': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_samples),
        'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Internet_Access': np.random.choice(['Yes', 'No'], n_samples),
        'Family_Income': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'School_Type': np.random.choice(['Public', 'Private'], n_samples),
        'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], n_samples),
        'Learning_Disabilities': np.random.choice(['Yes', 'No'], n_samples),
        'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], n_samples),
        'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic exam scores based on features
    score_base = (
        df['Hours_Studied'] * 1.2 +
        df['Attendance'] * 0.3 +
        df['Sleep_Hours'] * 2 +
        df['Previous_Scores'] * 0.4 +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    df['Exam_Score'] = score_base
    
    return df

@st.cache_resource
def train_model():
    """Train the regression model"""
    df = load_and_prepare_data()
    
    # Encode categorical variables
    categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                           'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                           'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                           'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    
    df_processed = df.copy()
    label_encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[feature] = le.fit_transform(df_processed[feature])
        label_encoders[feature] = le
    
    # Prepare features and target
    X = df_processed.drop('Exam_Score', axis=1)
    y = df_processed['Exam_Score']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, label_encoders, X.columns

def assign_grade(score):
    """Assign letter grade based on score"""
    if score >= 85:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 55:
        return 'C'
    else:
        return 'D'

def get_grade_color(grade):
    """Get color for grade visualization"""
    colors = {'A': '#00ff00', 'B': '#ffff00', 'C': '#ff8c00', 'D': '#ff0000'}
    return colors.get(grade, '#gray')

def create_gauge_chart(score, grade):
    """Create a gauge chart for the score"""
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
    """Create probability distribution chart"""
    # Simulate probability distribution around the predicted score
    x = np.linspace(max(0, score-20), min(100, score+20), 100)
    y = np.exp(-0.5 * ((x - score) / 5) ** 2)
    y = y / np.sum(y) * 100  # Convert to percentage
    
    fig = px.area(x=x, y=y, 
                  title="Prediction Confidence Distribution",
                  labels={'x': 'Possible Exam Scores', 'y': 'Probability (%)'},
                  color_discrete_sequence=['lightblue'])
    
    fig.add_vline(x=score, line_dash="dash", line_color="red", 
                  annotation_text=f"Predicted: {score:.1f}")
    
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    # Title and description
    st.title("🎓 Student Performance Predictor")
    st.markdown("### Prediksi nilai ujian berdasarkan berbagai faktor akademik dan personal")
    
    # Load model
    try:
        model, scaler, label_encoders, feature_columns = train_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Sidebar for inputs
    st.sidebar.header("📝 Input Fitur Siswa")
    st.sidebar.markdown("Masukkan informasi siswa untuk prediksi:")
    
    # Numerical inputs
    st.sidebar.subheader("📊 Data Numerik")
    hours_studied = st.sidebar.slider("Jam Belajar per Minggu", 0, 50, 15)
    attendance = st.sidebar.slider("Tingkat Kehadiran (%)", 0, 100, 80)
    sleep_hours = st.sidebar.slider("Jam Tidur per Hari", 3, 12, 7)
    previous_scores = st.sidebar.slider("Nilai Sebelumnya", 0, 100, 75)
    tutoring_sessions = st.sidebar.slider("Sesi Les per Bulan", 0, 10, 2)
    physical_activity = st.sidebar.slider("Aktivitas Fisik (jam/minggu)", 0, 20, 5)
    
    # Categorical inputs
    st.sidebar.subheader("📋 Data Kategorikal")
    parental_involvement = st.sidebar.selectbox("Keterlibatan Orang Tua", ['Low', 'Medium', 'High'])
    access_to_resources = st.sidebar.selectbox("Akses ke Sumber Belajar", ['Low', 'Medium', 'High'])
    extracurricular = st.sidebar.selectbox("Kegiatan Ekstrakurikuler", ['Yes', 'No'])
    motivation_level = st.sidebar.selectbox("Tingkat Motivasi", ['Low', 'Medium', 'High'])
    internet_access = st.sidebar.selectbox("Akses Internet", ['Yes', 'No'])
    family_income = st.sidebar.selectbox("Pendapatan Keluarga", ['Low', 'Medium', 'High'])
    teacher_quality = st.sidebar.selectbox("Kualitas Guru", ['Low', 'Medium', 'High'])
    school_type = st.sidebar.selectbox("Jenis Sekolah", ['Public', 'Private'])
    peer_influence = st.sidebar.selectbox("Pengaruh Teman", ['Positive', 'Neutral', 'Negative'])
    learning_disabilities = st.sidebar.selectbox("Kesulitan Belajar", ['Yes', 'No'])
    parental_education = st.sidebar.selectbox("Pendidikan Orang Tua", ['High School', 'College', 'Postgraduate'])
    distance_from_home = st.sidebar.selectbox("Jarak dari Rumah", ['Near', 'Moderate', 'Far'])
    gender = st.sidebar.selectbox("Jenis Kelamin", ['Male', 'Female'])
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Prediksi Hasil")
        
        if st.button("🔮 Prediksi Sekarang!", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'Hours_Studied': hours_studied,
                'Attendance': attendance,
                'Sleep_Hours': sleep_hours,
                'Previous_Scores': previous_scores,
                'Tutoring_Sessions': tutoring_sessions,
                'Physical_Activity': physical_activity,
                'Parental_Involvement': parental_involvement,
                'Access_to_Resources': access_to_resources,
                'Extracurricular_Activities': extracurricular,
                'Motivation_Level': motivation_level,
                'Internet_Access': internet_access,
                'Family_Income': family_income,
                'Teacher_Quality': teacher_quality,
                'School_Type': school_type,
                'Peer_Influence': peer_influence,
                'Learning_Disabilities': learning_disabilities,
                'Parental_Education_Level': parental_education,
                'Distance_from_Home': distance_from_home,
                'Gender': gender
            }
            
            # Encode categorical variables
            input_df = pd.DataFrame([input_data])
            
            categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                                   'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                                   'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                                   'Parental_Education_Level', 'Distance_from_Home', 'Gender']
            
            for feature in categorical_features:
                if feature in label_encoders:
                    try:
                        input_df[feature] = label_encoders[feature].transform(input_df[feature])
                    except ValueError:
                        # Handle unseen categories
                        input_df[feature] = 0
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            predicted_score = model.predict(input_scaled)[0]
            predicted_score = max(0, min(100, predicted_score))  # Clip to valid range
            grade = assign_grade(predicted_score)
            
            # Display results
            st.success(f"**Prediksi Nilai Ujian: {predicted_score:.1f}**")
            st.info(f"**Performance Grade: {grade}**")
            
            # Display fun message
            fun_message = random.choice(grade_messages[grade])
            st.markdown(f"### 🎉 {fun_message}")
            
            # Store results in session state for visualization
            st.session_state.predicted_score = predicted_score
            st.session_state.grade = grade
    
    with col2:
        st.subheader("📈 Visualisasi Hasil")
        
        if hasattr(st.session_state, 'predicted_score'):
            # Gauge chart
            gauge_fig = create_gauge_chart(st.session_state.predicted_score, st.session_state.grade)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Probability chart
            prob_fig = create_probability_chart(st.session_state.predicted_score)
            st.plotly_chart(prob_fig, use_container_width=True)
        else:
            st.info("Klik tombol 'Prediksi Sekarang!' untuk melihat visualisasi hasil")
    
    # Additional info
    st.markdown("---")
    st.subheader("📚 Informasi Sistem Penilaian")
    
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

if __name__ == "__main__":
    main()
