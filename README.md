🧠 AI Fake News Detection System
📌 Project Overview

The AI Fake News Detection System is a Machine Learning and Natural Language Processing (NLP) project that classifies news articles as Fake News or Real News.

The system performs text preprocessing, feature extraction using TF-IDF, and trains multiple machine learning models to achieve high classification accuracy.

A Streamlit web application is deployed for real-time prediction.

🎯 Objective

To build a text classification model that detects whether a news article is fake or real using Machine Learning and NLP techniques.

📂 Dataset

Dataset used: Fake and Real News Dataset (ISOT)
Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset contains:

Fake.csv → Fake news articles

True.csv → Real news articles

Total dataset size: ~44,000 news articles.

⚙️ Technologies Used

Python

Pandas & NumPy

Scikit-learn

Natural Language Processing (NLP)

TF-IDF Vectorization

Streamlit (Web Deployment)

Matplotlib

🧹 Data Preprocessing

The following preprocessing steps were performed:

Combined Fake and Real datasets

Added labels (Fake = 0, Real = 1)

Converted text to lowercase

Removed punctuation and special characters

Removed stopwords

Merged title and text into a single content column

This prepares the dataset for feature extraction and model training.

🔍 Feature Engineering

Used:

TF-IDF Vectorizer

N-gram features for improved performance

TF-IDF converts textual data into numerical form so machine learning models can process it.

🤖 Machine Learning Models

Multiple models were trained and compared:

Logistic Regression

Support Vector Machine (SVM)

Passive Aggressive Classifier

Ensemble Model (Best Performing Model)

📊 Model Performance
Model	Accuracy
Logistic Regression	~98%
SVM	~99%
Passive Aggressive	~99%
Ensemble Model	99.41%

The ensemble model achieved the highest validation accuracy.

📈 Evaluation Metrics

Model evaluation was performed using:

Accuracy

Confusion Matrix

Precision

Recall

F1-Score

These metrics ensure reliable performance assessment beyond simple accuracy.

🌐 Streamlit Web Application

A Streamlit web application was developed to allow real-time fake news detection.

Features:

User enters news text

Model processes and predicts

Displays result as Fake or Real

📄 Output Demonstration

The working output of the system is available in PDF format:

Real News Prediction 

<img width="961" height="488" alt="image" src="https://github.com/user-attachments/assets/9dc9fd94-65f0-4374-ab63-9b6a310c57c3" />


Fake News Prediction 

<img width="961" height="488" alt="image" src="https://github.com/user-attachments/assets/87154d95-092b-4d93-bb8d-f23f8981b53d" />


These demonstrate successful deployment and classification.

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
├── outputs/
│   ├── Output-1.pdf
│   └── Output-2.pdf
│
├── requirements.txt
└── README.md
▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run Streamlit Application
streamlit run app/app.py
🧪 Sample Input

Example:

The government announced new economic policies to support businesses.

Output:

Real News
