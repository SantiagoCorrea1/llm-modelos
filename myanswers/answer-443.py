import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def pipeline_mixto_regresion(df, target_col, cat_cols, num_cols):
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ],
        remainder='drop'
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    r2_test = float(pipeline.score(X_test, y_test))
    n_features_out = int(preprocessor.fit_transform(X_train).shape[1])

    return {
        'pipeline': pipeline,
        'r2_test': r2_test,
        'n_features_out': n_features_out,
    }