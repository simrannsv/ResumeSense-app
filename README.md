📄 ResumeSense
An AI-Powered Resume Classification System
ResumeSense is a deployed, end-to-end machine learning application that classifies resumes into predefined job categories using Natural Language Processing (NLP).
It leverages TF-IDF feature extraction and a Logistic Regression classifier to provide interpretable predictions along with confidence scores.

Live Demo
🔗 Deployed Application: https://resumesense-app-kdpkkup2ub5dfpaowrn238.streamlit.app/

 Key Features
Upload resumes in PDF format or paste resume text
Predicts resumes into 25 predefined job categories
Displays:
Predicted job category
Confidence score
Top 5 predictions
Probability distribution across all 25 categories
Clean and interactive Streamlit web interface
Fully deployed and accessible via browser

Machine Learning Pipeline
Raw Resume Text
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Numerical Feature Matrix
      ↓
Logistic Regression (Multi-class)
      ↓
Predicted Category + Probabilities


Tech Stack
Programming Language
Python
Libraries & Tools
scikit-learn
NumPy
Pandas
PyPDF2
Streamlit
ML Techniques
TF-IDF Vectorization
Logistic Regression (Supervised Learning)

Model Overview
Vectorizer: TF-IDF
Classifier: Logistic Regression
Problem Type: Multi-class text classification
Why Logistic Regression?
Performs well on high-dimensional sparse data
Fast, interpretable, and reliable
Strong baseline for NLP classification tasks
Note: Confidence scores may vary due to overlapping skill sets across job roles, which reflects real-world resume ambiguity.

Web Interface
The deployed Streamlit interface allows users to:
Upload or paste resume content
View predicted job category with confidence
Analyze Top 5 predictions
Inspect probabilities across all 25 job categories for transparency
This design improves interpretability rather than forcing a single hard label.

Project Structure
resume-classification/
│
├── app.py                  # Streamlit web application
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── trained_model.pkl       # Trained classification model
├── working_retrain.py      # Model training script
├── requirements.txt        # Project dependencies
└── README.md


How to Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Run the application
streamlit run app.py

Testing & Evaluation
The system was tested using resumes from multiple domains, including:
Data Science
Python Developer
Java Developer
DevOps Engineer
HR / Recruiter
Big Data (Hadoop)
The model distributes probabilities across relevant roles when skills overlap, which aligns with real-world resume patterns.

Learning Outcomes
This project provided hands-on experience in:
NLP-based text classification
Feature engineering with TF-IDF
Model evaluation and interpretation
Deploying ML models as web applications
Building explainable and user-focused ML systems

Simran Varthavani
AI/ML Aspirant
Interested in building practical, end-to-end machine learning systems
