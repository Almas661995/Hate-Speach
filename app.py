import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Get current date and time
current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%H:%M:%S")

# Add Custom CSS for Styling
st.markdown(f"""
    <style>
        .header {{
            background-color: #2c3e50; 
            color: #ecf0f1; 
            padding: 20px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }}

        .footer {{
            background-color: #1f1f1f; 
            color: white; 
            text-align: center; 
            padding: 15px; 
            position: fixed; 
            width: 100%; 
            bottom: 0; 
            left: 0;
        }}

        .footer .info {{
            font-size: 16px;
        }}

        .streamlit-container {{
            margin-bottom: 80px;
        }}
    </style>
    """, unsafe_allow_html=True)

# 🔥 App Title in Header
st.markdown(f'<div class="header">🔥 Hate Speech Detector <br> <span style="font-size:18px;">Date: {current_date}</span></div>', unsafe_allow_html=True)

# 📂 File Upload Section

st.write("## 📂 Upload Your Dataset (CSV)")
uploaded_file = st.file_uploader("Upload Your Dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load Dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Check for required columns
    if 'tweet' in data.columns and 'class' in data.columns:
        st.success("Dataset is valid! Ready for preprocessing. ⚙️")

        if st.button("⚙️ Preprocess Data", key="preprocess_data"):
            X = data['tweet']
            y = data['class']
            vectorizer = CountVectorizer(max_features=5000, stop_words='english')
            X_vectorized = vectorizer.fit_transform(X)
            st.write(f"Vectorized Matrix Shape: {X_vectorized.shape}")
            st.session_state['X_vectorized'] = X_vectorized
            st.session_state['y'] = y

        if 'X_vectorized' in st.session_state and 'y' in st.session_state:
            X_vectorized = st.session_state['X_vectorized']
            y = st.session_state['y']

            # 🎯 Algorithm Selection
            model_choice = st.selectbox(
                "Select a Classification Algorithm",
                ["Naive Bayes", "Logistic Regression", "Random Forest", "KNN", "Decision Tree"]
            )

            # ⚖️ Train-Test Split
            train_size = st.number_input("Enter Train Data Percentage (e.g., 80 for 80%)", min_value=1, max_value=99, value=80)
            train_ratio = train_size / 100

            if st.button("🚀 Train and Evaluate Model", key="train_model"):
                X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=1-train_ratio, random_state=42)

                # Model Initialization
                if model_choice == "Naive Bayes":
                    model = MultinomialNB()
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=200, multi_class='ovr')
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "KNN":
                    model = KNeighborsClassifier()
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)

                # Train the Model
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Save trained model
                st.session_state['model'] = model

                # Performance Metrics
                st.write("### 📊 Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                accuracy = accuracy_score(y_test, preds)
                st.write(f"### 🎯 Accuracy: {accuracy * 100:.2f}%")

                # 📉 Confusion Matrix
                st.write("### 📉 Confusion Matrix")
                cm = confusion_matrix(y_test, preds, labels=model.classes_)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

                # 📈 ROC Curve
                if hasattr(model, "predict_proba"):
                    st.write("### 📈 ROC Curve")
                    y_score = model.predict_proba(X_test)
                    fpr, tpr, roc_auc = {}, {}, {}
                    for i, label in enumerate(model.classes_):
                        fpr[label], tpr[label], _ = roc_curve(y_test == label, y_score[:, i])
                        roc_auc[label] = auc(fpr[label], tpr[label])
                    fig, ax = plt.subplots()
                    for label in model.classes_:
                        ax.plot(fpr[label], tpr[label], label=f'{label} (AUC = {roc_auc[label]:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend(loc='best')
                    st.pyplot(fig)

        # 📝 Real-Time Prediction
        st.write("### 🔥 Real-Time Tweet Classification")
        user_input = st.text_area("Enter a Tweet:")

        if st.button("🚀 Classify Tweet", key="classify_tweet"):
            if 'model' in st.session_state:
                vectorizer = CountVectorizer(max_features=5000, stop_words='english')
                vectorizer.fit(data['tweet'])
                input_vectorized = vectorizer.transform([user_input])
                model = st.session_state['model']
                prediction = model.predict(input_vectorized)[0]

                # Show Prediction with Icons
                if prediction == 0:
                    st.error(f"🔴 Prediction: **Hate Speech** ❌", icon="❌")
                elif prediction == 1:
                    st.warning(f"🟠 Prediction: **Offensive Language** ⚠️", icon="⚠️")
                else:
                    st.success(f"🟢 Prediction: **No Hate/Offensive Content** ✅", icon="✔️")
            else:
                st.warning("⚠️ Please preprocess and train the model first.")

# 📌 Footer with Session Info
st.markdown(f"""
    <div class="footer">
        <p class="info">
            <strong>Session:</strong> 2020-2024 | <strong>Class:</strong> BSCS-VIII <br>
            <strong>Developed by:</strong> Sharjeel Arshad & Awais Khan <br>
            <strong>Time:</strong> {current_time}
        </p>
    </div>
    """, unsafe_allow_html=True)
