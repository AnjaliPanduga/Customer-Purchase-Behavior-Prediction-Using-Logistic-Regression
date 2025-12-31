# 🚗 Customer Purchase Behavior Prediction Using Logistic Regression

An interactive Machine Learning project that predicts whether a customer will purchase a car based on **Age** and **Estimated Salary**, deployed as a **Streamlit web app**.

---

## 📌 Project Overview

This application uses **Logistic Regression** to perform binary classification on customer data and predict purchase behavior.  
Users can:
- Upload a dataset
- Train the model
- Evaluate performance
- Visualize decision boundaries
- Predict on new data

---

## 📂 Project Files

Customer-Purchase-Behavior-Prediction-Using-Logistic-Regression/
│
├── app.py # Streamlit Application Code
├── datasets/
│ ├── final1.csv # Training Dataset
│ └── logit classification.csv# Sample Dataset
├── requirements.txt # Dependencies
└── README.md # Project Documentation

## 🛠 Tech Stack

| Tool/Library | Purpose |
|-------------|---------|
| Python      | Programming Language |
| Streamlit   | Web App UX/UI |
| NumPy       | Numerical Operations |
| Pandas      | Data Handling |
| Matplotlib  | Plotting & Visualization |
| scikit-learn| Machine Learning |

---

## 🧠 How It Works

1. Upload your CSV dataset.
2. Features (**Age**, **Estimated Salary**) are selected.
3. Data is split into train & test sets.
4. Features are scaled.
5. Logistic Regression model is trained.
6. Model evaluation and performance are shown.
7. Predictions on new data.

## 📈 Key Features

✔ Upload training dataset  
✔ Train Logistic Regression classifier  
✔ Visualize model decision boundaries  
✔ Confusion matrix & classification report  
✔ ROC curve with AUC  
✔ Predict on new dataset  
✔ Download prediction results

## ▶️ How to Run Locally
1. Clone the Repo
```sh
git clone https://github.com/AnjaliPanduga/Customer-Purchase-Behavior-Prediction-Using-Logistic-Regression.git
cd Customer-Purchase-Behavior-Prediction-Using-Logistic-Regression
2. Install Dependencies
pip install -r requirements.txt
3. Run with Streamlit
streamlit run app.py
📄 Dataset Format
Training Dataset
Column	Description
Age	Customer Age
Estimated Salary	Customer Salary
Purchased	Target: 0 = Not Purchased, 1 = Purchased

📌 Notes
✔ Supported file types: CSV only
✔ Make sure Age & Salary columns are present

👩‍💻 Author
Anjali Panduga
📧 Email: pandugaanjali2003@gmail.com
🔗 GitHub: https://github.com/AnjaliPanduga
