# AI-Assistant-Satisfaction-Prediction-Engine
ML-based AI assistant satisfaction prediction engine with behavioral analytics and explainable insights.
🚀 AI Assistant Satisfaction Prediction Engine

A full-stack machine learning system that predicts user satisfaction (1–5 rating) for AI assistant interactions using behavioral data, feature engineering, and explainable AI techniques.

📌 Overview

This project builds an end-to-end ML pipeline to analyze how users interact with AI systems and predict their satisfaction levels. It combines behavioral analytics, supervised learning, and model explainability to uncover patterns that influence user experience.

🎓 Developed as part of an Honours DSA (Data Structures & Algorithms) Group Mini Project, this work demonstrates the practical application of analytical thinking, data processing, and algorithmic modeling in real-world AI systems.

🧠 Key Features
🔍 Behavioral feature extraction (device, usage type, session patterns)
🏗️ End-to-end ML pipeline using scikit-learn
🌲 Multi-class classification with Random Forest
📊 Model evaluation (accuracy, F1-score, confusion matrix)
💡 Explainable predictions using SHAP
🌐 Interactive dashboard built with Streamlit
⚙️ Ready-to-use pipeline for new data scoring
🗂️ Project Structure
ai-assistant-satisfaction-engine/
│
├── app.py                  # Streamlit dashboard
├── data/                  # Raw + processed datasets
├── models/                # Trained model
├── reports/               # Metrics & visualizations
├── src/                   # Core ML pipeline scripts
├── requirements.txt
└── README.md
📊 Dataset Summary

The dataset consists of AI interaction sessions with features like:

Device type (Mobile, Desktop, etc.)
Usage category (Coding, Writing, Research)
Session duration
Prompt length
Tokens used
AI model used
Timestamp → converted into time-based features

🎯 Target: User satisfaction rating (1–5)

⚙️ ML Pipeline

The pipeline includes:

Data preprocessing (handling categorical + numerical features)
Feature engineering (time-based signals)
Model training using Random Forest
Evaluation using classification metrics
Explainability with SHAP
📈 Key Insights
📱 Mobile usage tends to reduce satisfaction
🧑‍💻 Coding-related usage shows higher satisfaction
📅 Weekend interactions are generally more positive
🤖 Better AI models directly improve satisfaction
⏱️ Longer sessions → higher engagement → better ratings
💻 Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train the model
python -m src.train_model
3️⃣ Evaluate performance
python -m src.evaluate
4️⃣ Run dashboard
streamlit run app.py
🔮 Predict on New Data
python -m src.score_new_sessions path/to/input.csv

Outputs:

Predicted satisfaction rating
Probability distribution
🛠️ Tech Stack
Python
scikit-learn
pandas, numpy
matplotlib / seaborn
SHAP (Explainable AI)
Streamlit
⚠️ Limitations
Small dataset (synthetic)
Satisfaction is subjective
No user-level personalization
Limited temporal modeling
🚀 Future Improvements
Use ordinal regression
Add deep learning (LSTM / Transformers)
Deploy using FastAPI
Enhance dashboard with cohort analysis
Add real-world dataset integration
📌 Conclusion

This project demonstrates how machine learning can be used not just for prediction, but for understanding user behavior and improving AI systems. It combines technical rigor with practical insights, making it highly relevant for real-world AI product development.
