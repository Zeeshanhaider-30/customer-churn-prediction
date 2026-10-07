# Customer Churn Prediction

## 📌 Project Overview

This project uses Machine Learning to predict whether a customer is likely to **churn (leave)** or **stay** with a company.

The project includes data preprocessing, model training, model comparison, and evaluation using multiple performance metrics.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can identify customers who are likely to leave a service.

This can help businesses identify high-risk customers and take actions to improve customer retention.

## 📊 Dataset

The project uses a Telecom Customer Churn dataset containing customer information such as:

- Customer demographics
- Services used
- Contract information
- Payment method
- Monthly charges
- Total charges
- Customer churn information

## ⚙️ Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
```

## 🤖 Models Used

The following Machine Learning algorithms are used and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

## 🔍 Logistic Regression

Logistic Regression is used as a baseline classification model for predicting whether a customer will churn or stay.

## 🌲 Random Forest

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make predictions.

## 🚀 XGBoost

XGBoost is a powerful gradient boosting algorithm used for classification and is included to compare its performance with the other models.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── models/
│   └── best_model.pkl
│
├── data/
│   └── dataset
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project folder:

```bash
cd customer-churn-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open the churn prediction notebook and run the cells.

## 📌 Results

The trained models are compared using Accuracy, Precision, Recall, and F1-Score.

The model with the best overall performance is selected for the final prediction system.

## 👨‍💻 Author

**Zeeshan Haider**

Computer Science | Machine Learning | Data Analysis

## 📄 License

This project is created for educational and portfolio purposes.
