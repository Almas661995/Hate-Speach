import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Add custom CSS
st.markdown("""
    <style>
        /* Body Styling */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            color: #333;
        }

        /* Header Styling */
        .header {
            background-color: #1f77b4;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }

        /* Footer Styling */
        .footer {
            background-color: #333;
            color: white;
            text-align: center;
            padding: 10px;
            position: fixed;
            width: 100%;
            bottom: 0;
            left: 0;
        }

        .footer .date-time {
            font-size: 14px;
        }

        /* Button Styling */
        .btn {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }

        .btn:hover {
            background-color: #45a049;
        }

        /* Section Titles */
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #d62728;
            margin-bottom: 20px;
        }

        /* Sub-header Titles */
        .sub-header {
            font-size: 20px;
            font-weight: bold;
            color: #2ca02c;
        }

        /* Streamlit's Markdown Padding */
        .streamlit-container {
            margin-bottom: 50px;
        }
    </style>
    """, unsafe_allow_html=True)

# App Title in Header
st.markdown('<div class="header">Multiclass Hate Speech Detection System</div>', unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("Upload Your Dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load Dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Check for required columns
    if 'tweet' in data.columns and 'class' in data.columns:
        st.success("Dataset is valid! Ready for preprocessing.")

        if st.button("Preprocess Data", key="preprocess_data"):
            X = data['tweet']
            y = data['class']
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_tfidf = vectorizer.fit_transform(X)
            st.write(f"TF-IDF Matrix Shape: {X_tfidf.shape}")
            st.session_state['X_tfidf'] = X_tfidf
            st.session_state['y'] = y

        if 'X_tfidf' in st.session_state and 'y' in st.session_state:
            X_tfidf = st.session_state['X_tfidf']
            y = st.session_state['y']

            # Algorithm Selection
            model_choice = st.selectbox(
                "Select a Classification Algorithm",
                ["Naive Bayes", "Logistic Regression", "Random Forest", "KNN", "Decision Tree"]
            )

            # Train-Test Split
            train_size = st.number_input("Enter Train Data Percentage (e.g., 80 for 80%)", min_value=1, max_value=99, value=80)
            train_ratio = train_size / 100

            if st.button("Train and Evaluate Model", key="train_model"):
                X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=1-train_ratio, random_state=42)

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

                # Save the trained model in session state
                st.session_state['model'] = model

                # Performance Metrics
                st.write("<p class='section-header'>Classification Report</p>", unsafe_allow_html=True)
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                accuracy = accuracy_score(y_test, preds)
                precision = report['accuracy']
                recall = report['macro avg']['recall']
                f1_score = report['macro avg']['f1-score']
                st.write(f"### Accuracy: {accuracy * 100:.2f}%")
                st.write(f"### Precision: {precision * 100:.2f}%")
                st.write(f"### Recall: {recall * 100:.2f}%")
                st.write(f"### F1-Score: {f1_score * 100:.2f}%")

                # Confusion Matrix
                st.write("<p class='section-header'>Confusion Matrix</p>", unsafe_allow_html=True)
                cm = confusion_matrix(y_test, preds, labels=model.classes_)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

                # ROC Curve
                if hasattr(model, "predict_proba"):
                    st.write("<p class='section-header'>ROC Curve</p>", unsafe_allow_html=True)
                    y_score = model.predict_proba(X_test)
                    fpr = {}
                    tpr = {}
                    roc_auc = {}
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

        # Real-Time Prediction
        st.write("<p class='section-header'>Real-Time Tweet Classification</p>", unsafe_allow_html=True)
        user_input = st.text_area("Enter a Tweet:")

        if st.button("Classify Tweet", key="classify_tweet"):
            if 'model' in st.session_state:
                vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                vectorizer.fit(data['tweet'])
                input_tfidf = vectorizer.transform([user_input])
                model = st.session_state['model']  # Access the trained model
                prediction = model.predict(input_tfidf)[0]

                if prediction == 0:
                    st.error(f"Prediction: **Hate Speech** 😡", icon="❌")
                elif prediction == 1:
                    st.warning(f"Prediction: **Offensive Language** ⚠️", icon="⚠️")
                else:
                    st.success(f"Prediction: **No Hate/Offensive Content** ✅", icon="✔️")
            else:
                st.warning("Please preprocess and train the model first.")

# Footer with Date and Time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
    <div class="footer">
        <p>Multiclass Hate Speech Detection System | Last Updated: {current_time}</p>
    </div>
    """, unsafe_allow_html=True)
