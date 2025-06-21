# data_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

class CoralReefDataGenerator:
    def __init__(self):
        self.reef_locations = self._initialize_reef_database()
        self.environmental_baselines = self._calculate_environmental_baselines()

    def _initialize_reef_database(self) -> Dict:
        return {
            'Palmyra Atoll': {
                'lat': 5.8719, 'lon': -162.0864, 'region': 'Line Islands',
                'depth_range': (2, 50), 'ecosystem_type': 'Atoll',
                'baseline_temp': 27.8, 'seasonal_amplitude': 1.8,
                'upwelling_influence': 0.6, 'anthropogenic_stress': 0.1
            }
            # Add more reefs if needed
        }

    def _calculate_environmental_baselines(self) -> Dict:
        baselines = {}
        for name, data in self.reef_locations.items():
            baselines[name] = {
                'sst_baseline': data['baseline_temp'],
                'sst_std': data['seasonal_amplitude'] * 0.3,
                'coral_cover_baseline': 75 - data['anthropogenic_stress'] * 30
            }
        return baselines

    def generate_dataset(self) -> pd.DataFrame:
        np.random.seed(42)
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 6, 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        records = []
        for name, location in self.reef_locations.items():
            baseline = self.environmental_baselines[name]
            coral_cover = baseline['coral_cover_baseline']
            cumulative_stress = 0

            for date in dates:
                doy = date.timetuple().tm_yday
                seasonal = location['seasonal_amplitude'] * np.sin(2 * np.pi * doy / 365.25)
                temp = location['baseline_temp'] + seasonal + np.random.normal(0, 0.6)
                anomaly = temp - (location['baseline_temp'] + seasonal)
                dhw_contribution = anomaly / 7 if anomaly > 1.0 else 0
                cumulative_stress = cumulative_stress * 0.99 + dhw_contribution

                bleaching_severity = min(100, max(0, cumulative_stress * 15))
                coral_cover = max(5, coral_cover - bleaching_severity * 0.02 + np.random.normal(0, 0.3))

                records.append({
                    'reef_name': name,
                    'date': date,
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'sst': round(temp, 2),
                    'sst_anomaly': round(anomaly, 2),
                    'dhw': round(cumulative_stress, 2),
                    'coral_cover': round(coral_cover, 2),
                    'bleaching_severity': round(bleaching_severity, 1)
                })

        return pd.DataFrame(records)
