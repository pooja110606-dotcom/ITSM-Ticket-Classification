import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ----------------------------
# Data Splitting & Encoding
# ----------------------------

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
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


# ----------------------------
# Metric Calculation
# ----------------------------

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision(1)": precision_score(y_true, y_pred, pos_label=1),
        "Recall(1)": recall_score(y_true, y_pred, pos_label=1),
        "F1(1)": f1_score(y_true, y_pred, pos_label=1)
    }


# ----------------------------
# Logistic Regression
# ----------------------------

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


# ----------------------------
# Random Forest (GridSearch)
# ----------------------------

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='recall'
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


# ----------------------------
# XGBoost
# ----------------------------

def train_xgboost(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42
    )

    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 1, 3, 5],
        'min_child_weight': [1, 5, 10]
    }

    rs = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,
        scoring='recall',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


# ----------------------------
# CatBoost
# ----------------------------

def train_catboost(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weights = [1, neg / pos]

    cat = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Recall',
        class_weights=class_weights,
        verbose=0,
        random_state=42
    )

    param_dist = {
        'iterations': [300, 500, 800],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 10],
        'bagging_temperature': [0, 0.5, 1],
        'border_count': [64, 128]
    }

    rs_cat = RandomizedSearchCV(
        estimator=cat,
        param_distributions=param_dist,
        n_iter=20,
        scoring='recall',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    rs_cat.fit(X_train, y_train)
    return rs_cat.best_estimator_


# ----------------------------
# Model Evaluation Table
# ----------------------------

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = get_metrics(y_train, y_train_pred)
    test_metrics = get_metrics(y_test, y_test_pred)

    return [
        model_name,
        train_metrics["Accuracy"],
        train_metrics["Precision(1)"],
        train_metrics["Recall(1)"],
        train_metrics["F1(1)"],
        test_metrics["Accuracy"],
        test_metrics["Precision(1)"],
        test_metrics["Recall(1)"],
        test_metrics["F1(1)"]
    ]
