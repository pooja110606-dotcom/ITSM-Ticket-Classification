import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------
# Data Preparation
# --------------------------------

def prepare_assignment_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc,
        join='left',
        axis=1,
        fill_value=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    return X_train_scaled, X_test_scaled, y_train, y_test


# --------------------------------
# Logistic Regression
# --------------------------------

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


# --------------------------------
# Random Forest
# --------------------------------

def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    param_dist = {
        'n_estimators': [200, 400, 600],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2']
    }

    rs = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=25,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


# --------------------------------
# XGBoost
# --------------------------------

def train_xgboost(X_train, y_train):

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )

    param_dist = {
        'n_estimators': [300, 500, 800],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
        'gamma': [0, 1, 5]
    }

    rs = RandomizedSearchCV(
        xgb,
        param_dist,
        n_iter=25,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


# --------------------------------
# LightGBM
# --------------------------------

def train_lightgbm(X_train, y_train):

    lgb = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        class_weight='balanced',
        random_state=42
    )

    param_dist = {
        'n_estimators': [300, 500, 800],
        'max_depth': [-1, 10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 64, 100],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0]
    }

    rs = RandomizedSearchCV(
        lgb,
        param_dist,
        n_iter=25,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
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
        precision_score(y_train, y_train_pred, average='macro'),
        recall_score(y_train, y_train_pred, average='macro'),
        f1_score(y_train, y_train_pred, average='macro'),
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred, average='macro'),
        recall_score(y_test, y_test_pred, average='macro'),
        f1_score(y_test, y_test_pred, average='macro')
    ]
