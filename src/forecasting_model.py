import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from lightgbm import LGBMRegressor


# --------------------------------
# Prepare Daily Time Series
# --------------------------------

def prepare_daily_series(df, date_col, id_col):
    df[date_col] = pd.to_datetime(df[date_col])

    daily_df = (
        df
        .groupby(df[date_col].dt.date)
        .agg(incident_volume=(id_col, 'count'))
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date')

    full_dates = pd.date_range(
        start=daily_df['date'].min(),
        end=daily_df['date'].max()
    )

    daily_df = (
        daily_df
        .set_index('date')
        .reindex(full_dates, fill_value=0)
        .rename_axis('date')
        .reset_index()
    )

    return daily_df


# --------------------------------
# Train-Test Split
# --------------------------------

def train_test_split_ts(daily_df, split_ratio=0.8):
    split_date = daily_df['date'].quantile(split_ratio)

    train = daily_df[daily_df['date'] <= split_date]
    test = daily_df[daily_df['date'] > split_date]

    return train, test


# --------------------------------
# Naive Forecast
# --------------------------------

def naive_forecast(train, test):
    test = test.copy()
    test['naive_pred'] = test['incident_volume'].shift(1)
    test = test.dropna()

    mae = mean_absolute_error(
        test['incident_volume'],
        test['naive_pred']
    )

    return mae


# --------------------------------
# SARIMA
# --------------------------------

def train_sarima(train, test, order=(1,1,1), seasonal_order=(1,1,1,7)):
    model = SARIMAX(
        train['incident_volume'],
        order=order,
        seasonal_order=seasonal_order
    )
    fitted = model.fit()

    pred = fitted.predict(
        start=len(train),
        end=len(train) + len(test) - 1
    )

    mae = mean_absolute_error(
        test['incident_volume'],
        pred
    )

    return fitted, mae


# --------------------------------
# Prophet
# --------------------------------

def train_prophet(daily_df, test):
    prophet_df = daily_df[['date', 'incident_volume']].rename(
        columns={'date': 'ds', 'incident_volume': 'y'}
    )

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(prophet_df)

    prophet_test_df = test[['date']].rename(columns={'date': 'ds'})
    forecast = model.predict(prophet_test_df)

    mae = mean_absolute_error(
        test['incident_volume'],
        forecast['yhat']
    )

    return model, mae


# --------------------------------
# LightGBM Time Series
# --------------------------------

def train_lightgbm(daily_df, split_date):

    daily_df = daily_df.copy()
    daily_df['lag1'] = daily_df['incident_volume'].shift(1)
    daily_df['lag7'] = daily_df['incident_volume'].shift(7)
    daily_df = daily_df.dropna()

    train = daily_df[daily_df['date'] <= split_date]
    test = daily_df[daily_df['date'] > split_date]

    features = ['lag1', 'lag7']
    X_train, y_train = train[features], train['incident_volume']
    X_test, y_test = test[features], test['incident_volume']

    model = LGBMRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return model, mae


# --------------------------------
# 180-Day Forecasting
# --------------------------------

def future_forecast_180(
    daily_df,
    sarima_model,
    prophet_model,
    lgb_model,
    horizon=180
):

    future_dates = pd.date_range(
        start=daily_df['date'].max() + pd.Timedelta(days=1),
        periods=horizon
    )

    # Naive
    last_value = daily_df['incident_volume'].iloc[-1]
    naive = pd.Series([last_value] * horizon, index=future_dates)

    # SARIMA
    sarima_pred = sarima_model.get_forecast(steps=horizon).predicted_mean
    sarima_pred.index = future_dates

    # Prophet
    future_prophet = prophet_model.make_future_dataframe(periods=horizon)
    forecast = prophet_model.predict(future_prophet)
    prophet_pred = forecast[['ds', 'yhat']].tail(horizon)
    prophet_pred.set_index('ds', inplace=True)
    prophet_pred = prophet_pred['yhat']

    # LightGBM recursive
    last_row = daily_df.iloc[-1].copy()
    ml_preds = []

    for i in range(horizon):
        lag1 = last_row['incident_volume']
        lag7 = (
            daily_df.iloc[-7+i]['incident_volume']
            if i < 7 else ml_preds[i-7]
        )

        X_new = pd.DataFrame([[lag1, lag7]], columns=['lag1', 'lag7'])
        pred = lgb_model.predict(X_new)[0]

        ml_preds.append(pred)
        last_row['incident_volume'] = pred

    ml_series = pd.Series(ml_preds, index=future_dates)

    forecast_df = pd.DataFrame({
        'Naive': naive,
        'SARIMA': sarima_pred,
        'Prophet': prophet_pred,
        'LightGBM': ml_series
    })

    return forecast_df
