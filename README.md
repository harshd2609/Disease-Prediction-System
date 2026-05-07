# 🩺 Multi Disease Prediction System

A Machine Learning-based healthcare application that predicts:

* Diabetes
* Heart Disease
* Breast Cancer

using multiple classification algorithms and a Streamlit web interface.

---

# 🚀 Features

✅ Disease Prediction using ML Models
✅ Interactive Streamlit Web App
✅ Data Preprocessing & Feature Scaling
✅ Exploratory Data Analysis (EDA)
✅ Multiple Model Comparison
✅ ROC-AUC & Accuracy Evaluation
✅ Saved Models using `.pkl` files

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

# 📂 Project Structure

```bash
├── app.py
├── diabetes_model.pkl
├── heart_model.pkl
├── cancer_model.pkl
├── diabetes_scaler.pkl
├── heart_scaler.pkl
├── cancer_scaler.pkl
├── requirements.txt
├── disease_prediction.ipynb
└── README.md
```

---

# 📊 Machine Learning Models

The project uses multiple ML algorithms:

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)
* Gradient Boosting
* K-Nearest Neighbors (KNN)

Best-performing models are selected based on accuracy and ROC-AUC score.

---

# 📈 Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Scaling using StandardScaler
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Model Saving with Joblib
9. Streamlit Deployment

---

# ▶️ Run the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start Streamlit App

```bash
streamlit run app.py
```

---

# 🖥️ About `app.py`

`app.py` is the main Streamlit application file.

It:

* Loads trained `.pkl` models
* Takes user medical inputs
* Applies scaling
* Predicts diseases
* Displays prediction results interactively

---

# 💾 Model Files

`.pkl` files store trained Machine Learning models and scalers using Joblib.

Example:

```python
joblib.load('diabetes_model.pkl')
```

---

# 📌 Future Improvements

* Deploy on Streamlit Cloud
* Add User Authentication
* Improve UI Design
* Add More Disease Predictions
* Use Deep Learning Models

---

# 👨‍💻 Author

Harsh Diwakar
MCA (Data Science) | Gautam Buddha University
