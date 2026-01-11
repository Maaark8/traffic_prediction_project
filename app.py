import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- PAGE CONFIG ---
st.set_page_config(page_title="Traffic AI Project", layout="wide")

# --- 1. DATA LOADING (Cached so it doesn't reload every click) ---
@st.cache_data
def load_data():
    return pd.read_csv('data/traffic_clean.csv')

df = load_data()

# Prepare features (X) and target (y)
# We drop columns that aren't inputs (like the original text columns if any remain)
feature_cols = [c for c in df.columns if c not in ['Severity', 'is_severe', 'Weather_Condition', 'Sunrise_Sunset']]
X = df[feature_cols]
y = df['is_severe']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data (Crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Visualization", "Train Models", "Prediction Playground"])

# ==========================================
# PAGE 1: DATA VISUALIZATION (Class 3)
# ==========================================
if page == "Data Visualization":
    st.title("📊 Data Visualization & Analysis")
    st.markdown("Exploratory Data Analysis of **50,000 Traffic Accidents**.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accident Severity Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df['Severity'], hue=df['Severity'], palette='viridis', legend=False, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Geographic Heatmap")
        map_data = df[['Start_Lat', 'Start_Lng']].rename(columns={'Start_Lat': 'lat', 'Start_Lng': 'lon'})
        # Drop any empty rows just in case, to prevent map crashes
        st.map(map_data.dropna())

    st.subheader("Correlation Matrix")
    # specific numeric columns for correlation
    corr_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Severity']
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ==========================================
# PAGE 2: TRAIN MODELS (Class 1 & 2)
# ==========================================
elif page == "Train Models":
    st.title("🤖 Model Training Lab")
    
    # Create a models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    tab1, tab2 = st.tabs(["Logistic Regression (ML Class)", "Neural Network (Neural Comp Class)"])

    # --- TAB 1: LOGISTIC REGRESSION ---
    with tab1:
        st.header("Logistic Regression Config")
        c_val = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, key="lr_C")
        
        if st.button("Train Logistic Regression"):
            with st.spinner("Training Classical Model..."):
                lr_model = LogisticRegression(C=c_val, max_iter=1000)
                lr_model.fit(X_train_scaled, y_train)
                y_pred = lr_model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                
                # Save to Session State (for immediate use)
                st.session_state['lr_model'] = lr_model
                st.session_state['scaler'] = scaler
                st.session_state['model_columns'] = X_train.columns.tolist() # SAVE COLUMNS!
                
                st.success(f"Training Complete! Accuracy: {acc:.2%}")
                
                # Show Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

        # SAVE BUTTON (Only shows if model is trained)
        if 'lr_model' in st.session_state:
            if st.button("Save Logistic Model to Disk"):
                joblib.dump(st.session_state['lr_model'], 'models/logistic_model.pkl')
                joblib.dump(scaler, 'models/scaler.pkl') # Always save scaler with ML model
                joblib.dump(X_train.columns.tolist(), 'models/columns.pkl')
                st.toast("Logistic Model, Scaler, and Columns saved to /models/ folder!", icon="💾")

    # --- TAB 2: NEURAL NETWORK ---
    with tab2:
        st.header("Neural Network Config")
        
        epochs = st.slider("Epochs", 5, 50, 10)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        hidden_layers = st.slider("Hidden Neurons", 8, 128, 64)
        
        if st.button("Train Neural Network"):
            with st.spinner("Training..."):
                model = Sequential()
                model.add(Dense(hidden_layers, activation='relu', input_shape=(X_train_scaled.shape[1],)))
                model.add(Dropout(0.2))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                history = model.fit(X_train_scaled, y_train, 
                                    validation_data=(X_test_scaled, y_test),
                                    epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Save to Session State
                st.session_state['nn_model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['model_columns'] = X_train.columns.tolist()
                st.session_state['history'] = history.history
                
                st.success(f"Training Complete! Final Val Acc: {history.history['val_accuracy'][-1]:.2%}")
                
                # Plot History
                hist_df = pd.DataFrame(history.history)
                st.line_chart(hist_df[['accuracy', 'val_accuracy']])

        # SAVE BUTTON
        if 'nn_model' in st.session_state:
            if st.button("Save Neural Network to Disk"):
                st.session_state['nn_model'].save('models/neural_net.h5')
                joblib.dump(scaler, 'models/scaler.pkl')
                joblib.dump(X_train.columns.tolist(), 'models/columns.pkl')
                st.toast("Neural Network saved to /models/ folder!", icon="💾")

# ==========================================
# PAGE 3: PREDICTION (The "App" Part)
# ==========================================
elif page == "Prediction Playground":
    st.title("🔮 Interactive Prediction")
    
    # 1. SELECT MODEL
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Neural Network"])
    
    # 2. CHECK & LOAD RESOURCES (Model + Scaler + Columns)
    model = None
    scaler_loaded = None
    columns_loaded = None
    
    # Try loading from Session State first, then Disk
    try:
        if model_choice == "Logistic Regression":
            if 'lr_model' in st.session_state:
                model = st.session_state['lr_model']
                scaler_loaded = st.session_state['scaler']
                columns_loaded = st.session_state['model_columns']
            elif os.path.exists('models/logistic_model.pkl'):
                model = joblib.load('models/logistic_model.pkl')
                scaler_loaded = joblib.load('models/scaler.pkl')
                columns_loaded = joblib.load('models/columns.pkl')
                
        elif model_choice == "Neural Network":
            if 'nn_model' in st.session_state:
                model = st.session_state['nn_model']
                scaler_loaded = st.session_state['scaler']
                columns_loaded = st.session_state['model_columns']
            elif os.path.exists('models/neural_net.h5'):
                from tensorflow.keras.models import load_model
                model = load_model('models/neural_net.h5')
                scaler_loaded = joblib.load('models/scaler.pkl')
                columns_loaded = joblib.load('models/columns.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # 3. IF MODEL IS MISSING, STOP HERE
    if model is None or scaler_loaded is None:
        st.warning("⚠️ No model found! Please go to 'Train Models' and train (or save) a model first.")
        st.stop()

    st.success(f"Active Model: {model_choice}")

    # 4. USER INPUTS
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (F)", -20, 110, 70)
        humid = st.slider("Humidity (%)", 0, 100, 50)
        wind = st.slider("Wind Speed (mph)", 0, 100, 10)
    with col2:
        vis = st.slider("Visibility (miles)", 0, 10, 10)
        is_night = st.selectbox("Time of Day", ["Day", "Night"])
        weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Other"])

    # 5. PREDICTION LOGIC
    if st.button("Predict Severity Risk"):
        
        # A. Create a dictionary with the numeric inputs
        input_data = {
            'Temperature(F)': temp,
            'Humidity(%)': humid,
            'Visibility(mi)': vis,
            'Wind_Speed(mph)': wind,
            'Is_Night': 1 if is_night == "Night" else 0
        }
        
        # B. Handle One-Hot Encoding for Weather manually
        # This ensures we match the training columns exactly
        # Note: 'Clear' is often the baseline (all 0s) depending on drop_first=True
        possible_weathers = ["Cloudy", "Rain", "Snow", "Fog", "Other"]
        
        for w in possible_weathers:
            # Create column name like 'Weather_Simple_Rain'
            col_name = f"Weather_Simple_{w}"
            # Set to 1 if matched, else 0
            input_data[col_name] = 1 if weather == w else 0
            
        # C. Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # D. Align Columns (CRITICAL STEP)
        # Ensure input_df has exactly the same columns as X_train, in the same order
        # If any are missing (e.g. if 'Weather_Simple_Fog' wasn't in training data), fill with 0
        # If extra columns exist, drop them.
        input_df = input_df.reindex(columns=columns_loaded, fill_value=0)
        
        # E. Scale the Data
        input_scaled = scaler_loaded.transform(input_df)
        
        # F. Predict
        if model_choice == "Neural Network":
            prediction_prob = model.predict(input_scaled)[0][0]
            # Convert probability to class
            prediction_class = 1 if prediction_prob > 0.5 else 0
            confidence = prediction_prob if prediction_class == 1 else 1 - prediction_prob
        else:
            # Logistic Regression
            prediction_class = model.predict(input_scaled)[0]
            prediction_prob = model.predict_proba(input_scaled)[0][1] # Probability of class 1
            confidence = prediction_prob if prediction_class == 1 else 1 - prediction_prob

        # 6. DISPLAY RESULTS
        st.divider()
        if prediction_class == 1:
            st.error(f"🚨 Prediction: SEVERE Accident Risk")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.success(f"✅ Prediction: Minor Accident Risk")
            st.write(f"Confidence: {confidence:.2%}")
            
        st.info(f"Model used: {model_choice} | Raw Probability Score: {prediction_prob:.4f}")