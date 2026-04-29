import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('f1_features.csv')

FEATURES = [
    'lap_time_seconds', 'rolling_avg_lap_time', 'lap_time_delta',
    'TyreLife', 'tyre_age_squared', 'compound_encoded',
    'race_progress', 'laps_since_last_pit', 'stint_number',
    'position_norm', 'position_change', 'is_front_runner',
    'driver_encoded', 'team_encoded', 'racename_encoded'
]
TARGET = 'is_pit_lap'

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump((X_train_bal, X_test_scaled, y_train_bal, y_test, FEATURES), 'processed_data.pkl')

print(f"Training samples after SMOTE: {len(X_train_bal)}")
print(f"Test samples: {len(X_test_scaled)}")
print(f"Class balance after SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")