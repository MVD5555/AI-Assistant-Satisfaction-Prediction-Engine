# 🚀 AI Assistant Satisfaction Prediction Engine

A production-grade **machine learning system** that predicts user satisfaction (1–5 rating) for AI assistant interactions using behavioral data, feature engineering, and explainable AI techniques.

---

## 📌 Overview

This project implements an **end-to-end ML pipeline** to analyze user interaction patterns with AI systems and predict satisfaction levels. It combines **behavioral analytics, supervised learning, and explainable AI (XAI)** to uncover key drivers of user experience.

🎓 Developed as part of an **Honours DSA (Data Structures & Algorithms) Group Mini Project**, this work demonstrates practical application of data processing, algorithmic thinking, and machine learning in real-world AI systems.

---

## 🧠 Key Features

- 🔍 Behavioral feature extraction (device, usage type, session patterns)
- 🏗️ End-to-end ML pipeline using **scikit-learn**
- 🌲 Multi-class classification with **Random Forest**
- 📊 Model evaluation (Accuracy, Precision, Recall, F1-score)
- 💡 Explainable predictions using **SHAP**
- 🌐 Interactive analytics dashboard using **Streamlit**
- ⚙️ Scalable pipeline for **new data prediction**

---

## 🗂️ Project Structure


ai-assistant-satisfaction-engine/
│
├── app.py # Streamlit dashboard
├── data/ # Raw and processed datasets
├── models/ # Trained ML model
├── reports/ # Metrics and visualizations
├── src/ # Core ML pipeline scripts
│ ├── data_prep.py
│ ├── features.py
│ ├── train_model.py
│ ├── evaluate.py
│ ├── explain.py
│ └── score_new_sessions.py
│
├── requirements.txt
└── README.md


---

## 📊 Dataset Description

The dataset contains AI interaction sessions with the following features:

- Device type (Mobile, Desktop, Tablet, etc.)
- Usage category (Coding, Writing, Research, etc.)
- Prompt length
- Session duration
- Tokens used
- AI model used
- Timestamp (converted into time-based features)

🎯 **Target Variable:**  
User satisfaction rating (1–5)

---

## ⚙️ ML Pipeline

The pipeline consists of:

- Data preprocessing (categorical encoding + scaling)
- Feature engineering (time-based signals)
- Train-test split
- Model training using **Random Forest Classifier**
- Performance evaluation (classification metrics)
- Explainability using **SHAP values**

---

## 📈 Key Insights

- 📱 Mobile usage tends to reduce satisfaction  
- 🧑‍💻 Coding-related interactions show higher satisfaction  
- 📅 Weekend sessions are generally more positive  
- 🤖 Better AI models lead to higher satisfaction  
- ⏱️ Longer sessions correlate with better user experience  

---

## 💻 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the model
python -m src.train_model
3️⃣ Evaluate the model
python -m src.evaluate
4️⃣ Run the dashboard
streamlit run app.py
🔮 Predict on New Data
python -m src.score_new_sessions path/to/input.csv
Output:
Predicted satisfaction rating
Probability distribution across classes
🛠️ Tech Stack
Python
scikit-learn
pandas, numpy
matplotlib, seaborn
SHAP (Explainable AI)
Streamlit
⚠️ Limitations
Small synthetic dataset
Subjective nature of satisfaction
No user-level personalization
Limited temporal modeling
🚀 Future Enhancements
Implement ordinal regression
Add deep learning models (LSTM / Transformers)
Deploy as a FastAPI microservice
Improve dashboard with advanced analytics
Integrate real-world datasets
📌 Conclusion

This project demonstrates how machine learning can be leveraged not only for prediction but also for understanding user behavior and improving AI systems. It integrates modeling, explainability, and visualization into a cohesive solution, making it highly relevant for real-world AI applications.
