import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve

#-------- Global variables --------------
scaler = None
best_model = None
X = None
X_test = None
y_test = None

#---------- Data Preprocessing --------------

def Load_PreprocessData():
    global X, X_test, y_test, scaler

    df = pd.read_csv('diabetes.csv')

    columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_fix] = df[columns_to_fix].replace(0, np.nan)

    imputer = SimpleImputer(strategy='mean')
    df[columns_to_fix] = imputer.fit_transform(df[columns_to_fix])


    if 'Family History' not in df.columns:
        df['Family History'] = 0
    elif df['Family History'].dtype == object:
        df['Family History'] = df['Family History'].map({'Yes': 1, 'No': 0})

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test_local, y_train, y_test_local = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_test = X_test_local
    y_test = y_test_local
    return X_train, y_train

    # -------------Model Training--------------


def Train_Model(X_train, y_train):
    global best_model

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    best_auc = 0

    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_proba)
        print("ROC-AUC Score:", auc)
        
        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Save best model and scaler
    joblib.dump(best_model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# -------------Feature Importance + ROC Curve-----------------


def Visualizations():
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        features = X.columns
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances, y=features)
        plt.title("Feature Importance - Random Forest")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.show()

    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()


# ---------------User Interface GUI---------------

def launch_gui():
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler_local = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Model or scaler file not found.")
        return

    def predict_diabetes():
        try:
            values = [
                float(entry_pregnancies.get()),
                float(entry_glucose.get()),
                float(entry_bp.get()),
                float(entry_skin.get()),
                float(entry_insulin.get()),
                float(entry_bmi.get()),
                float(entry_dpf.get()),
                float(entry_age.get()),
                1 if entry_family.get().strip().lower() == "yes" else 0
            ]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return

        values_scaled = scaler_local.transform([values])
        prediction = model.predict(values_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        messagebox.showinfo("Prediction Result", f"The person is likely: {result}")

    # ------------------ GUI Layout ------------------

    app = tk.Tk()
    app.title("Diabetes Prediction")
    app.geometry("400x600")

    fields = [
        ("Pregnancies", "entry_pregnancies"),
        ("Glucose Level", "entry_glucose"),
        ("Blood Pressure", "entry_bp"),
        ("Skin Thickness", "entry_skin"),
        ("Insulin", "entry_insulin"),
        ("BMI", "entry_bmi"),
        ("Diabetes Pedigree Function", "entry_dpf"),
        ("Age", "entry_age"),
        ("Family History (Yes/No)", "entry_family")
    ]

    for label_text, var_name in fields:
        tk.Label(app, text=label_text).pack(pady=(10, 0))
        globals()[var_name] = tk.Entry(app)
        globals()[var_name].pack()

    tk.Button(app, text="Predict", command=predict_diabetes, bg="green", fg="white").pack(pady=20)
    app.mainloop()



def main():
    X_train, y_train = Load_PreprocessData()
    Train_Model(X_train, y_train)
    Visualizations()
    launch_gui()

if __name__ == "__main__":
    main()
