import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(
    page_title="📚 Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .grade-A { background-color: #28a745; color: white; padding: 10px; border-radius: 5px; }
    .grade-B { background-color: #17a2b8; color: white; padding: 10px; border-radius: 5px; }
    .grade-C { background-color: #ffc107; color: black; padding: 10px; border-radius: 5px; }
    .grade-D { background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; }
    .funny-quote {
        font-style: italic;
        font-size: 1.2rem;
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Funny quotes for each grade
funny_quotes = {
    'A': [
        "🌟 Wow! Kamu seperti Einstein yang sedang main-main di sekolah!",
        "🚀 Houston, we have a genius! Prestasi luar angkasa!",
        "👑 Raja/Ratu akademik telah muncul! Bow down peasants!",
        "🎯 Bullseye! Kamu menembak target dengan mata tertutup!",
        "🔥 So hot right now! Otak kamu sedang on fire!",
        "💎 Brilliant like a diamond! Terang benderang prestasi mu!"
    ],
    'B': [
        "👍 Solid performance! Kamu seperti Batman - reliable dan keren!",
        "🎵 Good vibes only! Prestasi yang bikin happy dance!",
        "🌈 Warna-warni prestasi! Not bad, not bad at all!",
        "⚡ Kamu seperti listrik - ada power nya tapi masih bisa distabilkan!",
        "🎪 Welcome to the show! Performance yang entertaining!",
        "🍕 Like a good pizza - satisfying dan bikin kenyang hati!"
    ],
    'C': [
        "🎢 Naik turun seperti roller coaster, tapi tetap seru!",
        "🐢 Slow and steady wins the race... eventually!",
        "☕ Butuh kopi double shot untuk boost performa!",
        "🌱 Masih tumbuh nih! Seperti tanaman yang butuh pupuk extra!",
        "🎭 Drama performa! Plot twist masih bisa terjadi!",
        "🚗 Mesin masih perlu tune-up, tapi masih jalan kok!"
    ],
    'D': [
        "💪 Rome wasn't built in a day! Semangat membangun kerajaan!",
        "🎯 Missing the target? Archer terbaik juga perlu latihan!",
        "🌟 Every expert was once a beginner! This is your origin story!",
        "🔧 Under construction! Masterpiece sedang dalam proses!",
        "🎮 Respawn time! Ready for the next level challenge!",
        "🌅 Setiap sunrise adalah kesempatan baru untuk shine!"
    ]
}

def assign_grade(score):
    """Assign performance grade based on exam score"""
    if score >= 85:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 55:
        return 'C'
    else:
        return 'D'

def get_random_quote(grade):
    """Get random funny quote based on grade"""
    return random.choice(funny_quotes[grade])

def create_mock_model():
    """Create a mock model for demonstration"""
    # This is a simplified model based on the analysis in your code
    # In real implementation, you would load the trained model
    model = LinearRegression()
    
    # Mock training (you should replace this with your actual trained model)
    np.random.seed(42)
    X_mock = np.random.randn(1000, 10)  # 10 features
    y_mock = (X_mock[:, 0] * 20 + X_mock[:, 1] * 15 + X_mock[:, 2] * 10 + 
              X_mock[:, 3] * 8 + X_mock[:, 4] * 5 + np.random.randn(1000) * 5 + 70)
    y_mock = np.clip(y_mock, 0, 100)
    
    model.fit(X_mock, y_mock)
    return model

# Load or create model
@st.cache_resource
def load_model():
    return create_mock_model()

@st.cache_resource
def load_scaler():
    return StandardScaler()

# Initialize model and scaler
model = load_model()
scaler = load_scaler()

# Main title
st.markdown('<h1 class="main-header">📚 Student Performance Predictor 🎓</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Prediksi nilai ujian berdasarkan berbagai faktor pembelajaran! 🚀
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for input features
st.sidebar.header("📝 Input Features")
st.sidebar.markdown("Masukkan informasi siswa di bawah ini:")

# Numerical features
st.sidebar.subheader("📊 Data Numerik")
hours_studied = st.sidebar.slider("📚 Hours Studied (per week)", 
                                 min_value=0, max_value=168, value=20, 
                                 help="Jam belajar per minggu")

attendance = st.sidebar.slider("📅 Attendance (%)", 
                              min_value=0, max_value=100, value=80,
                              help="Persentase kehadiran")

sleep_hours = st.sidebar.slider("😴 Sleep Hours (per day)", 
                               min_value=0, max_value=24, value=8,
                               help="Jam tidur per hari")

previous_scores = st.sidebar.slider("📈 Previous Scores", 
                                   min_value=0, max_value=100, value=75,
                                   help="Nilai sebelumnya")

tutoring_sessions = st.sidebar.slider("👨‍🏫 Tutoring Sessions (per month)", 
                                     min_value=0, max_value=30, value=5,
                                     help="Sesi bimbingan per bulan")

physical_activity = st.sidebar.slider("🏃‍♂️ Physical Activity (hours/week)", 
                                     min_value=0, max_value=168, value=5,
                                     help="Jam aktivitas fisik per minggu")

# Categorical features
st.sidebar.subheader("📋 Data Kategorikal")

parental_involvement = st.sidebar.selectbox("👨‍👩‍👧‍👦 Parental Involvement", 
                                          ['Low', 'Medium', 'High'])

access_to_resources = st.sidebar.selectbox("📚 Access to Resources", 
                                         ['Low', 'Medium', 'High'])

extracurricular = st.sidebar.selectbox("🎭 Extracurricular Activities", 
                                      ['Yes', 'No'])

motivation_level = st.sidebar.selectbox("🔥 Motivation Level", 
                                       ['Low', 'Medium', 'High'])

internet_access = st.sidebar.selectbox("🌐 Internet Access", 
                                      ['Yes', 'No'])

family_income = st.sidebar.selectbox("💰 Family Income", 
                                    ['Low', 'Medium', 'High'])

teacher_quality = st.sidebar.selectbox("👩‍🏫 Teacher Quality", 
                                      ['Low', 'Medium', 'High'])

school_type = st.sidebar.selectbox("🏫 School Type", 
                                  ['Public', 'Private'])

peer_influence = st.sidebar.selectbox("👥 Peer Influence", 
                                     ['Negative', 'Neutral', 'Positive'])

learning_disabilities = st.sidebar.selectbox("🧠 Learning Disabilities", 
                                            ['Yes', 'No'])

parental_education = st.sidebar.selectbox("🎓 Parental Education Level", 
                                        ['High School', 'College', 'Postgraduate'])

distance_from_home = st.sidebar.selectbox("🏠 Distance from Home", 
                                        ['Near', 'Moderate', 'Far'])

gender = st.sidebar.selectbox("👤 Gender", 
                             ['Male', 'Female'])

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Ready to Predict?")
    st.write("Pastikan semua data sudah diisi dengan benar di sidebar.")
    
    # Predict button
    if st.button("🔮 Predict Exam Score!", type="primary", use_container_width=True):
        # Prepare input data (simplified encoding for demo)
        # In real implementation, you should use the same encoders from training
        
        # Mock feature preparation (replace with actual feature engineering)
        features = np.array([[
            hours_studied, attendance, sleep_hours, previous_scores,
            tutoring_sessions, physical_activity,
            1 if parental_involvement == 'High' else 0,
            1 if access_to_resources == 'High' else 0,
            1 if motivation_level == 'High' else 0,
            1 if internet_access == 'Yes' else 0
        ]])
        
        # Make prediction
        predicted_score = model.predict(features)[0]
        predicted_score = max(0, min(100, predicted_score))  # Clip to 0-100 range
        
        # Get grade and quote
        grade = assign_grade(predicted_score)
        funny_quote = get_random_quote(grade)
        
        # Store in session state for displaying
        st.session_state.prediction = {
            'score': predicted_score,
            'grade': grade,
            'quote': funny_quote
        }

with col2:
    st.subheader("📊 Feature Summary")
    
    # Display input summary
    summary_data = {
        'Feature': ['Hours Studied', 'Attendance', 'Sleep Hours', 'Previous Scores', 
                   'Tutoring Sessions', 'Physical Activity'],
        'Value': [f"{hours_studied} hrs/week", f"{attendance}%", f"{sleep_hours} hrs/day",
                 f"{previous_scores}/100", f"{tutoring_sessions}/month", f"{physical_activity} hrs/week"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

# Display prediction results if available
if hasattr(st.session_state, 'prediction'):
    pred = st.session_state.prediction
    
    st.markdown("---")
    st.subheader("🎊 Prediction Results")
    
    # Main prediction display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>📈 Predicted Exam Score</h2>
            <h1 style="font-size: 4rem; margin: 1rem 0;">{pred['score']:.1f}</h1>
            <div class="grade-{pred['grade']}">
                <h3>Grade: {pred['grade']}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Funny quote
    st.markdown(f"""
    <div class="funny-quote">
        {pred['quote']}
    </div>
    """, unsafe_allow_html=True)
    
    # Additional info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Score Range", 
                 f"{max(0, pred['score']-5):.1f} - {min(100, pred['score']+5):.1f}",
                 help="Estimasi rentang nilai")
    
    with col2:
        st.metric("🎯 Performance Level", 
                 "Excellent" if pred['grade'] == 'A' else 
                 "Good" if pred['grade'] == 'B' else 
                 "Fair" if pred['grade'] == 'C' else "Needs Improvement")
    
    with col3:
        # Mock confidence (since we can't calculate real probability for manual grading)
        confidence = min(95, max(60, 75 + (pred['score'] - 50) * 0.4))
        st.metric("🎪 Prediction Confidence", f"{confidence:.1f}%",
                 help="Tingkat kepercayaan prediksi")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎓 Student Performance Predictor | Built with ❤️ using Streamlit</p>
    <p><em>Disclaimer: Ini adalah prediksi berdasarkan model machine learning. 
    Hasil aktual mungkin berbeda tergantung berbagai faktor lainnya.</em></p>
</div>
""", unsafe_allow_html=True)

# Add some educational content
with st.expander("📚 Tips untuk Meningkatkan Performa Akademik"):
    st.markdown("""
    ### 🚀 Tips Sukses Belajar:
    
    **📖 Manajemen Waktu Belajar:**
    - Buat jadwal belajar yang konsisten
    - Gunakan teknik Pomodoro (25 menit fokus, 5 menit istirahat)
    - Prioritaskan mata pelajaran yang sulit di waktu energi tinggi
    
    **🏫 Kehadiran & Partisipasi:**
    - Hadiri kelas secara teratur
    - Aktif bertanya dan berdiskusi
    - Catat poin-poin penting selama pembelajaran
    
    **😴 Kesehatan & Lifestyle:**
    - Tidur cukup 7-9 jam setiap malam
    - Olahraga teratur untuk menjaga kesehatan mental
    - Makan makanan bergizi untuk mendukung fungsi otak
    
    **👨‍👩‍👧‍👦 Dukungan Sosial:**
    - Komunikasi terbuka dengan orang tua
    - Bergabung dengan study group
    - Manfaatkan bimbingan belajar jika diperlukan
    """)

with st.expander("🔍 Tentang Model Prediksi"):
    st.markdown("""
    ### 🤖 Bagaimana Model Ini Bekerja?
    
    Model ini menggunakan **Linear Regression** untuk memprediksi nilai ujian berdasarkan berbagai faktor:
    
    **📊 Faktor Numerik:**
    - Jam belajar per minggu
    - Persentase kehadiran
    - Jam tidur per hari
    - Nilai sebelumnya
    - Sesi bimbingan belajar
    - Aktivitas fisik
    
    **📋 Faktor Kategorikal:**
    - Keterlibatan orang tua
    - Akses ke sumber belajar
    - Kegiatan ekstrakurikuler
    - Tingkat motivasi
    - Dan faktor lainnya...
    
    **⚠️ Catatan Penting:**
    Sistem grading (A, B, C, D) ditentukan secara manual berdasarkan rentang nilai:
    - A: 85-100 (Excellent)
    - B: 75-84 (Good) 
    - C: 55-74 (Fair)
    - D: 0-54 (Needs Improvement)
    """)