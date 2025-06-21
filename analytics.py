# analytics.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

class CoralAnalyticsEngine:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def _add_date_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date_numeric'] = pd.to_datetime(df['date']).astype(int) / 10**9
        return df

    def _calculate_trend(self, df: pd.DataFrame, column: str) -> dict:
        df = self._add_date_numeric(df)
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['date_numeric'], df[column])
        seconds_per_year = 365.25 * 24 * 3600
        annual_change = slope * seconds_per_year
        return {
            'slope': slope,
            'annual_change': annual_change,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def trend_summary(self, reef_name: str = None) -> dict:
        df = self.data[self.data['reef_name'] == reef_name] if reef_name else self.data
        return {
            'sst': self._calculate_trend(df, 'sst'),
            'dhw': self._calculate_trend(df, 'dhw'),
            'coral_cover': self._calculate_trend(df, 'coral_cover'),
            'bleaching_severity': self._calculate_trend(df, 'bleaching_severity')
        }

    def detect_bleaching_events(self, reef_name: str = None) -> pd.DataFrame:
        df = self.data[self.data['reef_name'] == reef_name] if reef_name else self.data
        df = df.sort_values(['reef_name', 'date'])
        events = []

        for reef in df['reef_name'].unique():
            reef_df = df[df['reef_name'] == reef]
            threshold_mask = reef_df['dhw'] > 4
            starts = reef_df[threshold_mask & ~threshold_mask.shift(1, fill_value=False)]
            ends = reef_df[threshold_mask & ~threshold_mask.shift(-1, fill_value=False)]

            for start, end in zip(starts['date'], ends['date']):
                segment = reef_df[(reef_df['date'] >= start) & (reef_df['date'] <= end)]
                events.append({
                    'reef': reef,
                    'start_date': start,
                    'end_date': end,
                    'duration': (end - start).days,
                    'max_dhw': segment['dhw'].max(),
                    'max_bleaching': segment['bleaching_severity'].max()
                })

        return pd.DataFrame(events)

    def predict_coral_cover(self) -> dict:
        features = ['sst', 'sst_anomaly', 'dhw']
        X = self.data[features]
        y = self.data['coral_cover']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }

        return results
