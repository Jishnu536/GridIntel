import streamlit as st
import pandas as pd
import os
import json
import tempfile
from main_pipeline import run_pipeline, plot_prediction_comparison, plot_prediction_scatter, plot_actual, plot_anomalies, plot_prediction

# Page Configuration
st.set_page_config(page_title="Rayfield Dashboard", layout="wide")
REMEMBER_FILE = "remember_me.json"

# Handle CSV upload globally so it's available across pages
st.markdown("### Energy Output Analysis and Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    st.session_state.csv_path = temp_file.name

# Session State Init
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "signin"
if "remember_me" not in st.session_state:
    st.session_state.remember_me = False

# Remember Me Utilities
def save_remember_me(username):
    with open(REMEMBER_FILE, "w") as f:
        json.dump({"username": username}, f)

def load_remember_me():
    if os.path.exists(REMEMBER_FILE):
        with open(REMEMBER_FILE, "r") as f:
            data = json.load(f)
            return data.get("username")
    return None

def clear_remember_me():
    if os.path.exists(REMEMBER_FILE):
        os.remove(REMEMBER_FILE)

# Page Navigation
def navigate_to(page_name):
    st.session_state.page = page_name

# Header
def display_header():
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.image("logo.png", width=50)
    with col_title:
        st.markdown("### **Rayfield Systems – AI-Powered Energy Automation**")
        st.caption("Automating Workflows for Energy Developers & Producers")

# Light Mode Overwrite
st.markdown("""
    <style>
        body, .main {
            background-color: white !important;
        }
        .icon-button {
            display: block;
            margin-bottom: 15px;
            background-color: #f0f0f0;
            padding: 12px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            cursor: pointer;
        }
        .sidebar-icons {
            position: fixed;
            right: 20px;
            top: 150px;
            z-index: 100;
        }
    </style>
""", unsafe_allow_html=True)

# Auto-login via Remember Me
if load_remember_me() and not st.session_state.authenticated:
    st.session_state.authenticated = True
    st.session_state.page = "csv_input"

# Sign-In Page
if not st.session_state.authenticated:
    display_header()
    st.markdown("---")

    col_center = st.columns([2, 4, 2])[1]
    with col_center:
        st.markdown("### Sign-In to Your Business Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember Me")

        if st.button("Sign In"):
            if username == "user" and password == "password123":
                st.session_state.authenticated = True
                st.session_state.page = "csv_input"
                st.session_state.remember_me = remember_me
                if remember_me:
                    save_remember_me(username)
                else:
                    clear_remember_me()
            else:
                st.error("Invalid credentials")

# Main App
else:
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        if st.button("➡️ CSV Input"):
            navigate_to("csv_input")
        if st.button("🏠 Home"):
            navigate_to("home")
        if st.button("⚠️ Anomalies"):
            navigate_to("Anomalies")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "signin"
            st.session_state.remember_me = False
            clear_remember_me()

    # Header
    col_logo, col_title, col_notes, col_menu = st.columns([1, 5, 2, 1])
    with col_logo:
        st.image("logo.png", width=50)
    with col_title:
        st.markdown("### **Rayfield Systems – AI-Powered Energy Automation**")
        st.caption("Automating Workflows for Energy Developers & Producers")
    with col_notes:
        st.button("Issues/Notes")

    st.markdown("---")

    # Page Routing
    def render_csv_input():
        st.markdown("### CSV Status")
        if "csv_path" in st.session_state:
            st.success("CSV uploaded and ready for processing!")
        else:
            st.warning("Please upload a CSV file")
        
        # Fallback default path
        default_path = "/Users/jishnu/Downloads/NREL NSRDB Datasets 2018-23/Cleaned Data/2023_data_cleaned.csv"
        csv_path = st.session_state.get("csv_path", default_path)
        
        st.markdown("### Actual Output Chart")
        if "csv_path" in st.session_state:
            # Run your pipeline
            X_train, X_test, y_train, y_test, y_pred, daily_ghi, next_date, next_day_prediction, plot_data = run_pipeline(csv_path)

            # Plot 1
            fig1 = plot_actual(y_test)
            st.pyplot(fig1)
        else:
            st.warning("Please upload a CSV file")
        st.empty()

    def render_home():
        st.markdown("### Actual Output vs Predicted Output")

        # Fallback default path
        default_path = "/Users/jishnu/Downloads/NREL NSRDB Datasets 2018-23/Cleaned Data/2023_data_cleaned.csv"
        csv_path = st.session_state.get("csv_path", default_path)

        if "csv_path" in st.session_state:
            # Run your pipeline
            X_train, X_test, y_train, y_test, y_pred, daily_ghi, next_date, next_day_prediction, plot_data = run_pipeline(csv_path)

            # Plot 2
            fig2 = plot_prediction_comparison(y_test, y_pred)
            st.pyplot(fig2)

            # Plot 3
            fig3 = plot_prediction_scatter(y_test, y_pred)
            st.pyplot(fig3)
        else:
            st.warning("Please upload a CSV file")

        st.empty()
        st.markdown("### Tomorrow’s Prediction")
        if "csv_path" in st.session_state:
            # Run your pipeline
            X_train, X_test, y_train, y_test, y_pred, daily_ghi, next_date, next_day_prediction, plot_data = run_pipeline(csv_path)

            # Plot 4
            fig4 = plot_prediction(next_date, next_day_prediction, plot_data)
            st.pyplot(fig4)
        else:
            st.warning("Please upload a CSV file")

        st.empty()
        st.markdown("### Alerts Table")
        alerts_csv = "alerts_table.csv"
        if os.path.exists(alerts_csv):
            alerts_df = pd.read_csv(alerts_csv)
            st.dataframe(alerts_df)
        else:
            st.warning("No alerts available yet.") 

    def render_anomalies():
        st.markdown("### Anomalies Chart")

        # Fallback default path
        default_path = "/Users/jishnu/Downloads/NREL NSRDB Datasets 2018-23/Cleaned Data/2023_data_cleaned.csv"
        csv_path = st.session_state.get("csv_path", default_path)

        if "csv_path" in st.session_state:
            # Run your pipeline
            X_train, X_test, y_train, y_test, y_pred, daily_ghi, next_date, next_day_prediction, plot_data = run_pipeline(csv_path)

            # Plot 5
            fig5 = plot_anomalies(y_test, y_pred, X_test)
            st.pyplot(fig5)
        else:
            st.warning("Please upload a CSV file")

        st.empty()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Alerts Table")
            alerts_csv = "alerts_table.csv"
            if os.path.exists(alerts_csv):
                alerts_df = pd.read_csv(alerts_csv)
                st.dataframe(alerts_df)
            else:
                st.warning("No alerts available yet.")
        with col2:
            st.markdown("#### Text Summary")
            summary_path = "csv_summary.txt"
            if os.path.exists(summary_path):
                with open(summary_path, "r") as file:
                    summary_text = file.read()
                st.text(summary_text)
            else:
                st.warning("Summary file not found. Please run the pipeline first.")

    # Page Dispatcher
    page_map = {
        "csv_input": render_csv_input,
        "home": render_home,
        "Anomalies": render_anomalies
    }
    page_func = page_map.get(st.session_state.page, render_csv_input)
    page_func()
