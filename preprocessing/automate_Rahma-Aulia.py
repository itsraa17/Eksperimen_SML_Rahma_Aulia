import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load dataset
def load_dataset(path: str):
    df = pd.read_csv(path)
    return df

# 2. Remove Outliers (IQR)
def remove_outliers_iqr(data, cols):
    """
    Menghapus outlier menggunakan metode IQR.
    """
    cleaned = data.copy()
    for col in cols:
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
    return cleaned


# 3. Preprocessing
def preprocess_weather(df):
    numeric_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # penanganan outlier
    for _ in range(3):
        df = remove_outliers_iqr(df, numeric_cols)

    # Encode label
    encoder = LabelEncoder()
    df['weather_encoded'] = encoder.fit_transform(df['weather'])

    # Scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split fitur dan label
    X = df[numeric_cols]
    y = df['weather_encoded']

    return df, X, y


# 4. Main
if __name__ == "__main__":
    print("=== Running automated preprocessing ===")

    raw_path = "seattle-weather.csv"
    output_path = "preprocessing/weather_preprocessed.csv"

    df_raw = load_dataset(raw_path)
    df_clean, X, y = preprocess_weather(df_raw)

    df_clean.to_csv(output_path, index=False)

    print(f"Preprocessing selesai!")
    print(f"Dataset bersih disimpan di: {output_path}")
    print(f"Jumlah baris akhir: {len(df_clean)}")
