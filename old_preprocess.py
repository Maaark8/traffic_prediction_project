# preprocess.py
import pandas as pd
import numpy as np

def clean_data():
    print("Loading huge dataset... (this might take a minute)")
    # Load only necessary columns to save memory
    cols = ['Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)', 
            'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 
            'Weather_Condition', 'Sunrise_Sunset']
    
    # Adjust the filename to match your downloaded Kaggle file
    df = pd.read_csv('data/US_Accidents_March23.csv', usecols=cols)
    
    print(f"Original shape: {df.shape}")

    # 1. Drop missing values
    df = df.dropna()
    
    # 2. Filter for a specific state (Optional: keeps data manageable)
    # df = df[df['State'] == 'CA'] 

    # 3. Create a binary target for Logistic Regression (Severe vs Non-Severe)
    # Severity in dataset is usually 1-4. Let's make 3 and 4 "Severe" (1)
    df['is_severe'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

    # 4. Simplify Weather (Top 5 conditions, others = 'Other')
    top_weather = df['Weather_Condition'].value_counts().nlargest(5).index
    df['Weather_Simple'] = df['Weather_Condition'].apply(lambda x: x if x in top_weather else 'Other')

    # 5. Encoding (Turn text into numbers for the Neural Network)
    # Convert Day/Night to 0/1
    df['Is_Night'] = df['Sunrise_Sunset'].map({'Day': 0, 'Night': 1})
    
    # One-Hot Encode Weather
    df = pd.get_dummies(df, columns=['Weather_Simple'], drop_first=True)

    # 6. Sample the data (For Streamlit performance)
    # 50k rows is enough for a Uni project demo. 3 million rows will freeze Streamlit.
    df_sample = df.sample(n=50000, random_state=42)
    
    print(f"Final shape for App: {df_sample.shape}")
    
    # Save to CSV
    df_sample.to_csv('data/traffic_clean.csv', index=False)
    print("Saved to data/traffic_clean.csv")

if __name__ == "__main__":
    clean_data()