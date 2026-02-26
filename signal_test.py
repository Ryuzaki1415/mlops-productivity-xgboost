import pandas as pd
from config import DATA_PATH, TARGET_COLUMN
from feature_engineering import create_features

df = pd.read_csv(DATA_PATH)
df = create_features(df)

corr = df.corr(numeric_only=True)[TARGET_COLUMN].sort_values(key=abs, ascending=False)

print("\nCorrelation with target:")
print(corr)