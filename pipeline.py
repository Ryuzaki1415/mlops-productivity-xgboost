from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from xgboost import XGBRegressor
from config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, DERIVED_COLUMNS, RANDOM_STATE

def build_pipeline():
    numeric_features = NUMERICAL_COLUMNS + DERIVED_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", FunctionTransformer(), numeric_features),   # passthrough
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), CATEGORICAL_COLUMNS),
        ]
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        seed=RANDOM_STATE,
        n_estimators=1000,           # early stopping will cut this
        max_depth=4,                 # shallower = less overfit on 3k rows
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.5,
        enable_categorical=True,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline