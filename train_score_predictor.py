# train_score_predictor.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("ODI_Match_Data.csv", low_memory=False)

# Feature Engineering
df['total_runs'] = df[['runs_off_bat', 'extras']].sum(axis=1)
df['over'] = df['ball'].astype(str).str.extract(r'(\d+)\.').astype(float)

# Filter for first innings only
df = df[df['innings'] == 1]

# Grouping and aggregating data by match till specific over
over_limit = 25  # You can vary this later for testing
df = df[df['over'] <= over_limit]

grouped = df.groupby('match_id').agg({
    'total_runs': 'sum',
    'over': 'max',
    'wicket_type': lambda x: x.notna().sum()
}).reset_index()

# Compute average run rate of last 5 overs (approximation)
df['run_rate'] = df['total_runs'] / (df['over'] + 0.1)
run_rate_last5 = df[df['over'] >= (over_limit - 5)].groupby('match_id')['run_rate'].mean().reset_index()
run_rate_last5.columns = ['match_id', 'run_rate_last5']

# Merge all features
final_df = grouped.merge(run_rate_last5, on='match_id')
final_df = final_df.rename(columns={
    'total_runs': 'current_score',
    'wicket_type': 'wickets_lost',
    'over': 'overs_completed'
})

# Target: Total match score
target_df = df.groupby('match_id')['total_runs'].sum().reset_index()
target_df.columns = ['match_id', 'final_score']

final_df = final_df.merge(target_df, on='match_id')

# Features and Labels
X = final_df[['current_score', 'wickets_lost', 'overs_completed', 'run_rate_last5']]
y = final_df['final_score']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Trained | RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Save model
joblib.dump(model, "xgb_score_predictor.pkl")
print("📦 Model saved as xgb_score_predictor.pkl")
