# tg_bot/feature_engineer.py
import pandas as pd
import numpy as np
from .eda_patterns import HOURLY_PATTERNS, WEEKLY_PATTERNS

class FeatureEngineer:
    def __init__(self):
        self.feature_template = self._create_feature_template()
    
    def _create_feature_template(self):
        """Создает шаблон признаков на основе EDA"""
        return {
            # Базовые признаки
            'Global_reactive_power': 0.1,
            'Voltage': 240.0,
            'Global_intensity': 2.5,
            'Sub_metering_1': 0.0,
            'Sub_metering_2': 0.0, 
            'Sub_metering_3': 0.0,
            
            # Временные признаки
            'hour': 12,
            'day_of_week': 2,
            'month': 5,
            'is_weekend': 0,
            
            # Лаги (будут заполнены динамически)
            'lag_2h_ago': 0.0,
            'lag_6h_ago': 0.0,
            'lag_12h_ago': 0.0,
            'lag_same_day_24h': 0.0,
            'lag_week_ago_168h': 0.0,
            
            # Скользящие статистики
            'rolling_mean_3h_past': 0.0,
            'rolling_mean_24h_past': 0.0,
            'rolling_std_24h_past': 0.0,
            'rolling_mean_7d_past': 0.0,
            
            # Циклические признаки
            'hour_sin': 0.0,
            'hour_cos': 0.0,
            'month_sin': 0.0,
            'month_cos': 0.0,
            'day_of_week_sin': 0.0,
            'day_of_week_cos': 0.0,
            
            # Бинарные признаки
            'is_early_morning': 0,
            'is_midday': 0,
            'is_evening_peak': 0,
            'is_morning_peak': 0,
            'is_night': 0,
            'is_deep_night': 0,
            'is_week_start': 0,
            'is_week_end': 0,
            'is_high_season': 0,
            'is_low_season': 0,
            
            # Отношения
            'kitchen_ratio': 0.0,
            'laundry_ratio': 0.0,
            'ac_heating_ratio': 0.0,
        }
    
    def create_features(self, hour, day_of_week, month):
        """Создает признаки для конкретного времени"""
        features = self.feature_template.copy()
        
        # Обновляем временные признаки
        features.update({
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': 1 if day_of_week >= 5 else 0,
        })
        
        # Заполняем лаги на основе EDA паттернов
        features.update(self._create_lags(hour, day_of_week, month))
        
        # Заполняем циклические признаки
        features.update(self._create_cyclic_features(hour, day_of_week, month))
        
        # Заполняем бинарные признаки
        features.update(self._create_binary_features(hour, month, day_of_week))
        
        # Вычисляем отношения
        features.update(self._calculate_ratios(features))
        
        return pd.DataFrame([features])
    
    def _create_lags(self, hour, day_of_week, month):
        """Создает лаги на основе EDA паттернов"""
        return {
            'lag_2h_ago': HOURLY_PATTERNS['mean'].get((hour - 2) % 24, 1.0),
            'lag_6h_ago': HOURLY_PATTERNS['mean'].get((hour - 6) % 24, 1.0),
            'lag_12h_ago': HOURLY_PATTERNS['mean'].get((hour - 12) % 24, 1.0),
            'lag_same_day_24h': HOURLY_PATTERNS['mean'].get(hour, 1.0),
            'lag_week_ago_168h': HOURLY_PATTERNS['mean'].get(hour, 1.0),
            
            'rolling_mean_3h_past': np.mean([
                HOURLY_PATTERNS['mean'].get((hour - 1) % 24, 1.0),
                HOURLY_PATTERNS['mean'].get((hour - 2) % 24, 1.0),
                HOURLY_PATTERNS['mean'].get((hour - 3) % 24, 1.0)
            ]),
            'rolling_mean_24h_past': np.mean(list(HOURLY_PATTERNS['mean'].values())),
            'rolling_std_24h_past': np.std(list(HOURLY_PATTERNS['mean'].values())),
            'rolling_mean_7d_past': np.mean(list(WEEKLY_PATTERNS['mean'].values())),
        }
    
    def _create_cyclic_features(self, hour, day_of_week, month):
        """Создает циклические признаки"""
        return {
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'month_sin': np.sin(2 * np.pi * month/12),
            'month_cos': np.cos(2 * np.pi * month/12),
            'day_of_week_sin': np.sin(2 * np.pi * day_of_week/7),
            'day_of_week_cos': np.cos(2 * np.pi * day_of_week/7),
        }
    
    def _create_binary_features(self, hour, month, day_of_week):
        """Создает бинарные признаки"""
        return {
            'is_early_morning': 1 if 4 <= hour <= 6 else 0,
            'is_midday': 1 if 10 <= hour <= 16 else 0,
            'is_evening_peak': 1 if 18 <= hour <= 22 else 0,
            'is_morning_peak': 1 if 7 <= hour <= 9 else 0,
            'is_night': 1 if hour <= 5 or hour >= 23 else 0,
            'is_deep_night': 1 if 1 <= hour <= 4 else 0,
            'is_week_start': 1 if day_of_week in [0, 1] else 0,
            'is_week_end': 1 if day_of_week in [4, 5] else 0,
            'is_high_season': 1 if month in [1, 2, 12] else 0,
            'is_low_season': 1 if month in [6, 7, 8] else 0,
        }
    
    def _calculate_ratios(self, features):
        """Вычисляет отношения для суб-счетчиков"""
        total_sub = features['Sub_metering_1'] + features['Sub_metering_2'] + features['Sub_metering_3']
        return {
            'kitchen_ratio': features['Sub_metering_1'] / (total_sub + 0.001),
            'laundry_ratio': features['Sub_metering_2'] / (total_sub + 0.001),
            'ac_heating_ratio': features['Sub_metering_3'] / (total_sub + 0.001),
        }