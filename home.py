import pickle
import streamlit as st

# Load the model and CountVectorizer
try:
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open('cv.pkl', 'rb') as f:
        cv = pickle.load(f)
except FileNotFoundError:
    st.error("Model or CountVectorizer file not found. Please upload 'model.pkl' and 'cv.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")
    st.stop()

# Add custom CSS for background, text styling, and header with logo
st.markdown(
    """
    <style>
        /* Background color */
        body {
            background-color: #f0f8ff;
        }
        /* Header with logo */
        .stApp {
            padding: 2rem;
            font-family: Arial, sans-serif;
        }
        .header {
            background-image: url('logo.png'); /* Path to the logo */
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            text-align: center;
            padding: 50px 0;
            color: #ffffff;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: bold;
        }
        /* Style the markdown text */
        .stMarkdown {
            color: #34495e;
            font-size: 1.1rem;
        }
        /* Style text area */
        textarea {
            border: 1px solid #95a5a6;
            border-radius: 5px;
            padding: 10px;
            font-size: 1rem;
        }
        /* Style buttons */
        button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }
        /* Prediction styling */
        .success {
            color: #27ae60;
            font-weight: bold;
        }
        .warning {
            color: #e67e22;
            font-weight: bold;
        }
        .error {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title with logo background
st.markdown('<div class="header"><h1>Hate Speech Detection App</h1></div>', unsafe_allow_html=True)

# App description
st.markdown(
    """
    This app detects whether a given tweet contains:
    - **Hate Speech**
    - **Offensive Language**
    - **No Hate and Offensive Content**
    
    Enter a tweet in the text box below and click outside the box or press enter to see the result.
    """,
    unsafe_allow_html=True,
)

# User input
user = st.text_area("Enter a Tweet:")

# Prediction
if len(user.strip()) > 0:
    # Transform the input using the loaded CountVectorizer
    sample = user.strip()
    data = cv.transform([sample])

    # Predict using the loaded model
    prediction = clf.predict(data)

    # Display the prediction with styled output
    if prediction[0] == "Hate Speech":
        st.error(f"Prediction: **{prediction[0]}** 😡", icon="❌")
    elif prediction[0] == "Offensive Language":
        st.warning(f"Prediction: **{prediction[0]}** ⚠️", icon="⚠️")
    else:
        st.success(f"Prediction: **{prediction[0]}** ✅", icon="✔️")
else:
    st.info("Please enter some text to analyze.")

# Footer
st.markdown(
    """
    <hr style='border: 1px solid #95a5a6;'>
    <p style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
        This app uses a trained machine learning model to classify tweets. For best results, input clean and concise text.
    </p>
    """,
    unsafe_allow_html=True,
)
