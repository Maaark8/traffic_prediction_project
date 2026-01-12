# US Traffic Accident Severity Predictor

## Project Overview
This project is a comprehensive analysis and prediction tool for US Traffic Accidents. It leverages a massive dataset of traffic records to predict whether an accident will be **Minor** or **Severe** based on weather conditions and road infrastructure.

The project is built as an interactive **Streamlit Web Application** that demonstrates:
1.  **Big Data Visualization:** Interactive geospatial mapping and exploratory data analysis.
2.  **Classical Machine Learning:** Logistic Regression and Random Forest classifiers.
3.  **Neural Computing:** A deep learning Multi-Layer Perceptron (MLP) with real-time hyperparameter tuning.

---

## Features by Domain
*   **Geospatial Analysis:** Interactive map plotting accident hotspots (Latitude/Longitude).
*   **Correlation Heatmaps:** Visualizing relationships between weather variables (Wind, Temperature, Visibility).
*   **Distribution Plots:** Analyzing the frequency of accidents by severity and weather conditions.
*   **Logistic Regression:** A baseline linear model for binary classification.
*   **Random Forest:** An ensemble method utilizing decision trees to capture non-linear relationships and feature importance.
*   **Neural Network:** A Feed-Forward Neural Network (Keras/TensorFlow).
*   **Live Hyperparameter Tuning:** Users can adjust **Epochs**, **Batch Size**, and **Hidden Neurons** inside the app and retrain the model in real-time.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Maaark8/traffic_prediction_project
cd traffic-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Setup (Crucial Step)
The original dataset is >3GB. You must run the preprocessing script to clean, balance, and sample the data for the web app.
1.  Download `US_Accidents_March23.csv` from Kaggle.
2.  Place it in the `data/` folder.
3.  Run the script:
```bash
python preprocess.py
```
*This will create `data/traffic_clean.csv`, a balanced dataset ready for training.*

### 4. Run the Application
```bash
streamlit run app.py
```

---

## Project Structure

```text
traffic_prediction/
│
├── app.py                   # The Main Streamlit Application
├── preprocess.py            # Data Cleaning, Balancing (Undersampling), and Encoding
├── requirements.txt         # Project Dependencies
│
├── data/
│   ├── US_Accidents_Raw.csv # (Excluded from Git) Raw large dataset
│   └── traffic_clean.csv    # Processed, balanced dataset used by App
│
└──models/                  # Saved models & scalers (Generated automatically)
    ├── logistic_model.pkl
    ├── neural_net.h5
    ├── rf_model.pkl
    └── scaler.pkl

```

---

## Methodology

### 1. Data Preprocessing
*   **Class Balancing:** The raw dataset is heavily imbalanced (mostly minor accidents). Used **Undersampling** to create a 50/50 split between Minor and Severe cases to prevent model bias.
*   **Scaling:** Applied `StandardScaler` to normalize numerical inputs (Temperature, Wind Speed) for the Neural Network.

### 2. Model Architecture
*   **Logistic Regression:** Standard implementation with L2 regularization.
*   **Random Forest:** 50-100 trees with max depth control.
*   **Neural Network:**
    *   Input Layer: 14 Features (Scaled)
    *   Hidden Layer: ReLU activation + Dropout (0.2)
    *   Output Layer: Sigmoid activation (Binary Classification)

---

## Results & Observations
*   **Weather vs. Infrastructure:** We found that weather alone (Temperature, Wind, etc.) is a weak predictor of severity (~55% accuracy). Adding road features (Crossing, Junction, Traffic Signals) improved performance.
*   **Model Performance:**
    *   *Logistic Regression:* ~60% Accuracy
    *   *Neural Network:* ~64% Accuracy 
    *   *Random Forest:* ~70% Accuracy 

---

## Credits
*   **Dataset:** [US Accidents (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) via Kaggle.
*   **Frameworks:** Streamlit, TensorFlow, Scikit-Learn, Pandas, Plotly.