import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve
)

from matplotlib.colors import ListedColormap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Logistic Regression Classification App",
    page_icon="📊",
    layout="wide"
)

st.title("🚗 Logistic Regression – Car Purchase Prediction")
st.markdown("Predict whether a user will purchase a car based on **Age** and **Estimated Salary**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Dataset")

train_file = st.sidebar.file_uploader(
    "Upload Training Dataset (CSV)",
    type=["csv"]
)

# ---------------- MAIN LOGIC ----------------
if train_file is not None:
    dataset = pd.read_csv(train_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(dataset.head())

    # Feature & Target
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, -1].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train Model
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # ---------------- RESULTS ----------------
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix")
        st.dataframe(cm)

    with col2:
        st.write("Classification Report")
        cr = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(cr).transpose())

    # ---------------- BIAS & VARIANCE ----------------
    st.subheader("⚖️ Bias & Variance")
    st.write(f"Training Accuracy (Bias): {classifier.score(X_train, y_train):.2f}")
    st.write(f"Testing Accuracy (Variance): {classifier.score(X_test, y_test):.2f}")

    # ---------------- DECISION BOUNDARY (TRAINING) ----------------
    st.subheader("🧠 Decision Boundary – Training Set")

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )

    fig1, ax1 = plt.subplots()
    ax1.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green"))
    )

    for i, j in enumerate(np.unique(y_set)):
        ax1.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=ListedColormap(("red", "green"))(i),
            label=j
        )

    ax1.set_title("Logistic Regression (Training Set)")
    ax1.set_xlabel("Age (Scaled)")
    ax1.set_ylabel("Estimated Salary (Scaled)")
    ax1.legend()

    st.pyplot(fig1)

    # ---------------- DECISION BOUNDARY (TEST) ----------------
    st.subheader("🧪 Decision Boundary – Test Set")

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )

    fig2, ax2 = plt.subplots()
    ax2.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green"))
    )

    for i, j in enumerate(np.unique(y_set)):
        ax2.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=ListedColormap(("red", "green"))(i),
            label=j
        )

    ax2.set_title("Logistic Regression (Test Set)")
    ax2.set_xlabel("Age (Scaled)")
    ax2.set_ylabel("Estimated Salary (Scaled)")
    ax2.legend()

    st.pyplot(fig2)

    # ---------------- ROC CURVE ----------------
    st.subheader("📈 ROC Curve")

    y_pred_prob = classifier.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax3.plot([0, 1], [0, 1], linestyle="--")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend()
    ax3.grid()

    st.pyplot(fig3)

    # ---------------- NEW DATA PREDICTION ----------------
    st.subheader("🧪 Predict on New Dataset")

    new_file = st.file_uploader(
        "Upload New Dataset (CSV)",
        type=["csv"],
        key="newdata"
    )

    if new_file is not None:
        new_data = pd.read_csv(new_file)
        st.dataframe(new_data.head())

        d2 = new_data.copy()
        new_X = new_data.iloc[:, [3, 4]].values
        new_X = sc.transform(new_X)

        d2["Prediction"] = classifier.predict(new_X)

        st.success("Predictions completed!")
        st.dataframe(d2.head())

        csv = d2.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predicted CSV",
            csv,
            "predicted_output.csv",
            "text/csv"
        )

else:
    st.info("👈 Upload a training CSV file to begin")
