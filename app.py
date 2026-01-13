import streamlit as st
import pickle
import re
from docx import Document
import PyPDF2
import io
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ResumeSense",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 3rem 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-item {
        margin: 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #495057;
        font-size: 0.95rem;
    }
    
    .sidebar-item strong {
        color: #2c3e50;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .format-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        margin: 0.3rem 0;
        color: #495057;
    }
    
    .header-container {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 3rem;
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    .stTabs {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: white;
        border-radius: 10px;
        color: #495057;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 2rem;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .result-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(79, 172, 254, 0.4);
        transition: transform 0.3s ease;
        margin: 1rem 0;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .category-label {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .top5-section {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .category-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        border-left: 5px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .category-item:nth-child(1) { border-left-color: #667eea; }
    .category-item:nth-child(2) { border-left-color: #764ba2; }
    .category-item:nth-child(3) { border-left-color: #f093fb; }
    .category-item:nth-child(4) { border-left-color: #4facfe; }
    .category-item:nth-child(5) { border-left-color: #43e97b; }
    
    .category-item:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    .category-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .category-prob {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .rank-number {
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2.5rem;
        font-weight: 800;
        opacity: 0.1;
        color: #667eea;
    }
    
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
        margin: 2rem 0;
    }
    
    div[data-testid="stExpander"] {
        background: white;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
    }
    
    .uploadedFile {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found.")
        return None, None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None

# Sidebar content
def create_sidebar(model, vectorizer):
    with st.sidebar:
        st.markdown("### 📄 About")
        st.markdown("""
        <div class="sidebar-section">
            <p style="color: #495057; line-height: 1.6;">
                This AI-powered system classifies resumes into job categories using:
            </p>
            <div class="sidebar-item">
                🧠 <strong>Model:</strong> Logistic Regression
            </div>
            <div class="sidebar-item">
                📊 <strong>Features:</strong> TF-IDF (1000 features)
            </div>
            <div class="sidebar-item">
                🎯 <strong>Accuracy:</strong> ~98%
            </div>
        """, unsafe_allow_html=True)
        
        # Get categories from model
        if model is not None:
            num_categories = len(model.classes_)
            st.markdown(f"""
            <div class="sidebar-item">
                📂 <strong>Categories:</strong> {num_categories} job roles
            </div>
        </div>
        """, unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### 📈 Model Statistics")
        
        # Get stats from model and vectorizer
        if model is not None and vectorizer is not None:
            num_categories = len(model.classes_)
            num_features = len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else vectorizer.max_features
            
            st.markdown(f"""
            <div class="sidebar-section">
                <div style="text-align: center; padding: 1rem 0;">
                    <div class="stat-label">Total Categories</div>
                    <div class="stat-number">{num_categories}</div>
                </div>
                <div style="text-align: center; padding: 1rem 0;">
                    <div class="stat-label">Features Used</div>
                    <div class="stat-number">{num_features}</div>
                </div>
                <div style="text-align: center; padding: 1rem 0;">
                    <div class="stat-label">Training Samples</div>
                    <div class="stat-number">962</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📁 Supported Formats")
        st.markdown("""
        <div class="sidebar-section">
            <div class="format-item">✅ PDF (.pdf)</div>
            <div class="format-item">✅ Word (.docx)</div>
            <div class="format-item">✅ Text (.txt)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Categories")
        
        # Get categories dynamically from model
        if model is not None:
            categories = sorted(model.classes_)
            
            with st.expander("View All Categories"):
                for idx, cat in enumerate(categories, 1):
                    st.markdown(f"{idx}. {cat}")

def main():
    model, vectorizer = load_model()
    
    create_sidebar(model, vectorizer)
    
    # Header
    st.markdown("""
        <div class="header-container">
            <div class="main-title">📄 ResumeSense</div>
            <div class="subtitle">An AI-powered Resume Classification Tool</div>
        </div>
    """, unsafe_allow_html=True)
    
    if model is None or vectorizer is None:
        st.warning("⚠️ Please ensure your model files are properly loaded.")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["📤 Upload Resume", "📝 Paste Text"])
    
    uploaded_file = None
    pasted_text = ""
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        pasted_text = st.text_area(
            "Paste resume text here",
            height=300,
            placeholder="Copy and paste the resume content here..."
        )
    
    # Close the tabs section before displaying results
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button (outside tabs)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Analyze Resume", type="primary")
    
    if predict_button:
        resume_text = None
        
        if uploaded_file is not None:
            with st.spinner("📄 Extracting text from file..."):
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    resume_text = extract_text_from_txt(uploaded_file)
        elif pasted_text.strip():
            resume_text = pasted_text
        else:
            st.warning("⚠️ Please upload a file or paste resume text")
            return
        
        if resume_text:
            with st.spinner("🤖 Analyzing resume with AI..."):
                processed_text = preprocess_text(resume_text)
                text_vectorized = vectorizer.transform([processed_text])
                prediction = model.predict(text_vectorized)[0]
                
                try:
                    probabilities = model.predict_proba(text_vectorized)[0]
                    confidence = max(probabilities) * 100
                except:
                    confidence = None
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="success-message">✅ Analysis Complete!</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="category-label">Predicted Category</div>
                            <div class="prediction-value">{prediction}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if confidence:
                        st.markdown(f"""
                            <div class="result-card">
                                <div class="category-label">Confidence Score</div>
                                <div class="confidence-value">{confidence:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Top 5
                if confidence:
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    try:
                        classes = model.classes_
                        probs = probabilities * 100
                        
                        pred_df = pd.DataFrame({
                            'Category': classes,
                            'Probability': probs
                        })
                        
                        pred_df = pred_df.sort_values('Probability', ascending=False)
                        top5_df = pred_df.head(5).reset_index(drop=True)
                        
                        st.markdown('<div class="top5-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">🎯 Top 5 Predictions</div>', unsafe_allow_html=True)
                        
                        # Chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=top5_df['Category'],
                                y=top5_df['Probability'],
                                marker=dict(
                                    color=top5_df['Probability'],
                                    colorscale=[[0, '#667eea'], [0.5, '#764ba2'], [1, '#f093fb']],
                                    line=dict(width=0)
                                ),
                                text=[f"{p:.2f}%" for p in top5_df['Probability']],
                                textposition='outside',
                                textfont=dict(size=15, color='#2c3e50', family='Poppins'),
                                hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>'
                            )
                        ])
                        
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title="Probability (%)",
                            showlegend=False,
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=14, color='#2c3e50', family='Poppins'),
                            yaxis=dict(gridcolor='#e9ecef', gridwidth=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        for idx, row in top5_df.iterrows():
                            st.markdown(f"""
                                <div class="category-item">
                                    <span class="rank-number">{idx + 1}</span>
                                    <span class="category-name">{row['Category']}</span>
                                    <span class="category-prob">{row['Probability']:.2f}%</span>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error displaying predictions: {str(e)}")
                
                # Text preview
                st.markdown("<br><br>", unsafe_allow_html=True)
                with st.expander("📄 View Extracted Resume Text"):
                    preview_text = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
                    st.text_area("Resume Content", preview_text, height=250, disabled=True)

if __name__ == "__main__":
    main()

    