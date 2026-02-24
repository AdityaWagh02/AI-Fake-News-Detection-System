🧠 AI Fake News Detection System
<img width="1231" height="4" alt="image" src="https://github.com/user-attachments/assets/9a2970c4-69ea-4efd-ad42-c89c65f31a5a" />

📌 Project Overview

The AI Fake News Detection System is a Machine Learning and NLP-based project designed to classify news articles as Fake News or Real News. The system uses text preprocessing, TF-IDF feature extraction, and multiple machine learning models to achieve high accuracy. A Streamlit web application is deployed for real-time prediction.
<img width="1231" height="4" alt="image" src="https://github.com/user-attachments/assets/f6d5c01e-8285-4e20-a929-7aff8ad31653" />

🎯 Objective

To build an intelligent text classification model that can automatically detect whether a news article is fake or real using Natural Language Processing and Machine Learning techniques.
<img width="1231" height="4" alt="image" src="https://github.com/user-attachments/assets/90738878-a508-472f-a562-36c0ca8849a8" />

📂 Dataset

Dataset used: Fake and Real News Dataset (ISOT)
Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset contains:

Fake.csv → Fake news articles
True.csv → Real news articles
Total samples: ~44,000 news articles
<img width="1231" height="4" alt="image" src="https://github.com/user-attachments/assets/9a443f1a-3ec5-4763-8d7a-d015f3df94c1" />

⚙️ Technologies Used

Python
Machine Learning (Scikit-learn)
Natural Language Processing (NLP)
TF-IDF Vectorization
Streamlit (Web App)
Pandas, NumPy, Matplotlib
GitHub (Version Control)
<img width="1231" height="4" alt="image" src="https://github.com/user-attachments/assets/cdfc92ec-f609-4baf-afe0-bee7a083d115" />

🧹 Data Preprocessing

Steps performed:
Combined Fake and Real datasets
Added labels (Fake = 0, Real = 1)
Converted text to lowercase
Removed punctuation and special characters
Removed stopwords
Prepared cleaned dataset for feature extraction
<img width="1231" height="4" alt="image" src="https://github.com/user-attachments/assets/6b1a27c7-b412-4cfd-b667-f21b8369b987" />

🔍 Feature Engineering

Used:
TF-IDF Vectorizer
N-gram features (improves accuracy)

Purpose:
Convert text into numerical form so machine learning models can understand it.

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/b5bdf99c-1fde-4bb7-bc18-30ee7b4613a7" />

🤖 Machine Learning Models Used

We trained and compared multiple models:
Logistic Regression
Support Vector Machine (SVM)
Passive Aggressive Classifier
Ensemble Model (Best Model)

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/ddb912fb-bb1d-4669-83df-0bc957218611" />

📊 Model Performance
Model	Accuracy
Logistic Regression	~98%
SVM	~99%
Passive Aggressive	~99%
Ensemble Model	99.41%

Best model: Ensemble Model

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/0ccc1e99-99bd-4354-a56e-be04dcb45132" />

📈 Evaluation Metrics

Used evaluation techniques:
Accuracy Score
Confusion Matrix
Precision
Recall
F1-Score

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/57d6d6af-2371-49b5-aa22-c060aec5904d" />

🌐 Web Application (Streamlit)

We developed a web app for real-time fake news detection.
Features:
User enters news text
Model predicts Fake or Real
Instant results

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/537081cb-caa0-4f2d-9f34-8c3fec938c60" />

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

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/2ac36863-b455-48bb-9ff0-ae31d697e36f" />

▶️ How to Run the Project
Step 1: Install dependencies
pip install -r requirements.txt
Step 2: Run Streamlit app
streamlit run app/app.py

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/1394201c-68dd-4faa-a299-07a0c7dbf953" />

🧪 Sample Input

Example: The government said it will introduce new economic measures to improve growth and reduce unemployment.
Output: Real News

<img width="788" height="3" alt="image" src="https://github.com/user-attachments/assets/e5abd6a7-b019-4a03-92bd-d6bb3b6f6080" />

## 📄 Output Screenshots

<img width="961" height="488" alt="image" src="https://github.com/user-attachments/assets/cb82f265-22c1-4364-926c-b1ee40a0956a" />

<img width="961" height="488" alt="image" src="https://github.com/user-attachments/assets/8e1fc3ff-da15-4985-9c61-3d1e12cb6b75" />

These outputs demonstrate that the model successfully classifies news articles using the deployed Streamlit web application.
