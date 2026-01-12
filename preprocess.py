import pandas as pd
import numpy as np

def clean_data():
    print("Loading dataset...")
    
    cols = ['Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)', 
            'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 
            'Weather_Condition', 'Sunrise_Sunset',
            'Traffic_Signal', 'Crossing', 'Junction']
    
    df = pd.read_csv('data/US_Accidents_March23.csv', usecols=cols)

    df = df.dropna()
    
    for col in ['Traffic_Signal', 'Crossing', 'Junction']:
        df[col] = df[col].astype(int)

    df['is_severe'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

    print("Balancing classes...")
    df_severe = df[df['is_severe'] == 1]
    df_minor = df[df['is_severe'] == 0]
    
    min_len = min(len(df_severe), len(df_minor))
    sample_size = min(min_len, 50000) 
    
    df_severe_bal = df_severe.sample(n=sample_size, random_state=42)
    df_minor_bal = df_minor.sample(n=sample_size, random_state=42)
    
    df = pd.concat([df_severe_bal, df_minor_bal])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    top_weather = ['Clear', 'Cloudy', 'Overcast', 'Rain', 'Snow', 'Fog']
    def map_weather(w):
        w = str(w).lower()
        if 'snow' in w or 'wint' in w: return 'Snow'
        if 'rain' in w or 'storm' in w: return 'Rain'
        if 'fog' in w or 'mist' in w: return 'Fog'
        if 'cloud' in w or 'overcast' in w: return 'Cloudy'
        if 'clear' in w or 'fair' in w: return 'Clear'
        return 'Other'

    df['Weather_Simple'] = df['Weather_Condition'].apply(map_weather)
    df['Is_Night'] = df['Sunrise_Sunset'].map({'Day': 0, 'Night': 1})
    
    # One-Hot Encode
    df = pd.get_dummies(df, columns=['Weather_Simple'], drop_first=False, dtype=int)

    df_final = df.select_dtypes(include=[np.number])
    df_final.to_csv('data/traffic_clean.csv', index=False)
    print(f"Saved balanced data with Road Features. Shape: {df_final.shape}")

if __name__ == "__main__":
    clean_data()