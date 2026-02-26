from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, DERIVED_COLUMNS, RANDOM_STATE


def build_pipeline():
    
    print("Building Pipeline!")

    numeric_features = NUMERICAL_COLUMNS + DERIVED_COLUMNS

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline