# Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn or stay. It includes a leakage-safe training and evaluation workflow, model tuning, and an interactive Streamlit app.

## Dataset and workflow

The training data is `Telco_customer_churn.xlsx`. Customer identifiers, churn labels/scores/reasons, and other non-predictive location fields are excluded from the model features.

The raw feature data is split into training and test sets with `stratify=y` before any preprocessing is fitted. A scikit-learn `ColumnTransformer` imputes and scales numeric features and imputes and one-hot encodes nominal categories inside each model pipeline. Logistic Regression is a baseline; Random Forest and XGBoost use five-fold stratified cross-validation for model selection and hyperparameter tuning.

The selected model is evaluated against the held-out test set using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

## Setup (PowerShell)

```powershell
git clone https://github.com/Zeeshanhaider-30/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
python train.py
```

The script reports cross-validation results and held-out test metrics, then saves the complete preprocessing/model pipeline to `best_model.joblib`. The Jupyter notebook provides the same workflow:

```powershell
jupyter notebook churn_prediction.ipynb
```

## Run the app

```powershell
streamlit run app.py
```

The app reports the predicted Churn/Stay class, churn probability, and confidence in the predicted class. Run the training script first if the exported model is missing.

## Project structure

```text
customer-churn-prediction/
├── app.py                       # Streamlit prediction interface
├── train.py                     # Training, CV tuning, evaluation, and model export
├── churn_prediction.ipynb       # Notebook walkthrough using the shared workflow
├── Telco_customer_churn.xlsx    # Source training data
├── best_model.joblib            # Exported fitted pipeline
├── Project_Documentation.pdf    # Project documentation
├── requirements.txt             # Python dependencies
└── .gitignore
```

## Author

Zeeshan Haider

## License

Created for educational and portfolio purposes.
