🩺 AI-Based Multi-Disease Prediction System : 
A machine learning-powered system that predicts the probability of multiple diseases using general health checkup data and lifestyle factors. This project focuses on early risk assessment to support preventive healthcare.

🚀 Overview : 
This project leverages machine learning to analyze health parameters such as blood pressure, cholesterol, BMI, and lifestyle habits to predict the likelihood of diseases like:
Diabetes
Hypertension
Heart Disease
Anemia
Kidney Disease
:. The system provides probabilistic outputs (%), helping users understand their health risks before symptoms become severe.

🎯 Problem Statement : 
Early detection of chronic diseases is often missed due to lack of awareness and underutilization of routine health data. This project addresses:
Delayed diagnosis
Lack of preventive insights
Fragmented disease analysis
Limited accessibility to predictive tools

💡 Solution : 
A complete ML pipeline that:
Processes health and lifestyle data
Performs multi-disease probability prediction
Provides interpretable results through a user-friendly GUI

🧠 Features : 
✅ Multi-disease prediction (single input → multiple outputs)
✅ Uses real-world health indicators
✅ Feature engineering (BMI, Pulse Pressure, MAP, Lifestyle Risk)
✅ Model selection & hyperparameter tuning
✅ Explainable AI using SHAP
✅ Interactive GUI using Streamlit
✅ Scalable and deployment-ready

🛠️ Tech Stack : 
Programming Language: Python
Libraries:
pandas, numpy
scikit-learn
xgboost
joblib
shap
Visualization: matplotlib, seaborn
Frontend: Streamlit

📊 Machine Learning Workflow : 
Data Collection → Data Cleaning → EDA → Feature Engineering
→ Model Training → Model Evaluation → Hyperparameter Tuning
→ Model Export → GUI Development → Deployment

⚙️ Model Details : 
Algorithm Used: XGBoost Regressor (MultiOutput)
Evaluation Metrics:
Mean Squared Error (MSE)
R² Score
Why XGBoost?
Handles non-linear relationships
High performance and efficiency
Works well with structured/tabular data

📈 Sample Output : 
{
  "Diabetes": 87.23,
  "Hypertension": 91.11,
  "Heart Disease": 76.54,
  "Anemia": 5.12,
  "Kidney Disease": 64.89
}

🖥️ GUI Preview : 
The Streamlit app allows users to:
Input health parameters
Get instant predictions
View risk levels with visual indicators

Run the app locally:
streamlit run app.py

📦 Installation : 
1. Clone the repository
git clone https://github.com/your-username/multi-disease-prediction.git
cd multi-disease-prediction
2. Install dependencies
pip install -r requirements.txt

🔍 Real-World Use Cases : 
Early health risk assessment
Preventive healthcare planning
Fitness & wellness applications
Clinical pre-screening tools

📌 Future Improvements : 
Add SHAP explanations in GUI
Generate downloadable health reports (PDF)
Deploy on cloud (Streamlit Cloud / AWS / Render)
Integrate with real-world medical datasets

⚠️ Disclaimer : 
This project is for educational and research purposes only.
It is not a substitute for professional medical advice.

👨‍💻 Author : 
Abhishek Gorya

⭐ Support : 
If you found this project useful, consider giving it a ⭐ on GitHub!
