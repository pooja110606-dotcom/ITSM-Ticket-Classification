import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------
# Feature Engineering
# --------------------------------

def prepare_change_features(df):

    df = df.copy()

    df['Open_Time'] = pd.to_datetime(df['Open_Time'], dayfirst=True, errors='coerce')
    df['Resolved_Time'] = pd.to_datetime(df['Resolved_Time'], dayfirst=True, errors='coerce')
    df['Close_Time'] = pd.to_datetime(df['Close_Time'], dayfirst=True, errors='coerce')

    df['resolution_hours'] = np.where(
        df['Resolved_Time'].notna(),
        (df['Resolved_Time'] - df['Open_Time']).dt.total_seconds() / 3600,
        (df['Close_Time'] - df['Open_Time']).dt.total_seconds() / 3600
    )

    max_hours = df['resolution_hours'].quantile(0.95)
    df['resolution_hours'] = df['resolution_hours'].fillna(max_hours)

    df['rfc_generated'] = (df['No_of_Related_Changes'] > 0).astype(int)

    leakage_cols = [
        'Incident_ID',
        'Related_Change',
        'Related_Interaction',
        'Closure_Code',
        'Resolved_Time',
        'Close_Time'
    ]

    df = df.drop(columns=leakage_cols, errors='ignore')

    return df


# --------------------------------
# Frequency Encoding
# --------------------------------

def frequency_encode(X):

    X = X.copy()

    cat_cols = X.select_dtypes(include=['object']).columns

    for col in cat_cols:
        freq = X[col].value_counts(normalize=True)
        X[col] = X[col].map(freq)

    X[cat_cols] = X[cat_cols].fillna(0)

    return X


# --------------------------------
# Train-Test Split (Time-based)
# --------------------------------

def time_split(X, y, df):

    df = df.sort_values('Open_Time')
    split = int(0.8 * len(df))

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test


# --------------------------------
# Scaling
# --------------------------------

def scale_features(X_train, X_test):

    scaler = StandardScaler()

    num_cols = X_train.select_dtypes(exclude=['object']).columns

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test


# --------------------------------
# Models
# --------------------------------

def train_logistic(X_train, y_train):

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rs = RandomizedSearchCV(
        rf,
        params,
        n_iter=20,
        scoring='recall',
        cv=3,
        n_jobs=-1
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


def train_xgboost(X_train, y_train):

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.9, 1],
        'colsample_bytree': [0.7, 0.9, 1]
    }

    rs = RandomizedSearchCV(
        xgb,
        params,
        n_iter=20,
        scoring='recall',
        cv=3,
        n_jobs=-1
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


def train_lightgbm(X_train, y_train):

    lgb = LGBMClassifier(
        class_weight='balanced',
        random_state=42
    )

    params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 64, 128],
        'subsample': [0.7, 0.9, 1]
    }

    rs = RandomizedSearchCV(
        lgb,
        params,
        n_iter=20,
        scoring='recall',
        cv=3,
        n_jobs=-1
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


# --------------------------------
# Evaluation
# --------------------------------

def evaluate_model(model, name, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return [
        name,
        accuracy_score(y_train, y_train_pred),
        precision_score(y_train, y_train_pred),
        recall_score(y_train, y_train_pred),
        f1_score(y_train, y_train_pred),
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred)
    ]
