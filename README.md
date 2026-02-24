🧠 AI Fake News Detection System

📌 Project Overview
The AI Fake News Detection System is a Machine Learning and NLP-based project designed to classify news articles as Fake News or Real News. The system uses text preprocessing, TF-IDF feature extraction, and multiple machine learning models to achieve high accuracy. A Streamlit web application is deployed for real-time prediction.

🎯 Objective
To build an intelligent text classification model that can automatically detect whether a news article is fake or real using Natural Language Processing and Machine Learning techniques.

📂 Dataset
Dataset used: Fake and Real News Dataset (ISOT)
Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset contains:
Fake.csv → Fake news articles
True.csv → Real news articles
Total samples: ~44,000 news articles

⚙️ Technologies Used
Python
Machine Learning (Scikit-learn)
Natural Language Processing (NLP)
TF-IDF Vectorization
Streamlit (Web App)
Pandas, NumPy, Matplotlib
GitHub (Version Control)

🧹 Data Preprocessing

Steps performed:
Combined Fake and Real datasets
Added labels (Fake = 0, Real = 1)
Converted text to lowercase
Removed punctuation and special characters
Removed stopwords
Prepared cleaned dataset for feature extraction

🔍 Feature Engineering

Used:
TF-IDF Vectorizer
N-gram features (improves accuracy)

Purpose:
Convert text into numerical form so machine learning models can understand it.

🤖 Machine Learning Models Used

We trained and compared multiple models:
Logistic Regression
Support Vector Machine (SVM)
Passive Aggressive Classifier
Ensemble Model (Best Model)

📊 Model Performance
Model	Accuracy
Logistic Regression	~98%
SVM	~99%
Passive Aggressive	~99%
Ensemble Model	99.41%

Best model: Ensemble Model

📈 Evaluation Metrics

Used evaluation techniques:
Accuracy Score
Confusion Matrix
Precision
Recall
F1-Score

🌐 Web Application (Streamlit)

We developed a web app for real-time fake news detection.
Features:
User enters news text
Model predicts Fake or Real
Instant results

📁 Project Structure
AI-Fake-News-Detection-System/
│
├── Dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
│   └── fake_news_detection.ipynb
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md

▶️ How to Run the Project
Step 1: Install dependencies
pip install -r requirements.txt
Step 2: Run Streamlit app
streamlit run app/app.py

🧪 Sample Input

Example: The government said it will introduce new economic measures to improve growth and reduce unemployment.
Output: Real News

## 📄 Output Screenshots

<img width="961" height="488" alt="image" src="https://github.com/user-attachments/assets/cb82f265-22c1-4364-926c-b1ee40a0956a" />
<img width="1001" height="488" alt="image" src="https://github.com/user-attachments/assets/8e1fc3ff-da15-4985-9c61-3d1e12cb6b75" />

These outputs demonstrate that the model successfully classifies news articles using the deployed Streamlit web application.
