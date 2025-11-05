import pandas as pd
import numpy as np
from datetime import datetime
import json

class SimpleFeatureGenerator:
    def __init__(self):
        # РЕАЛЬНЫЕ данные из твоего EDA
        self.hourly_avg = {
            0: 0.778, 1: 0.634, 2: 0.540, 3: 0.517, 4: 0.489, 5: 0.527,
            6: 0.940, 7: 1.518, 8: 1.492, 9: 1.340, 10: 1.200, 11: 1.102,
            12: 1.054, 13: 1.000, 14: 1.040, 15: 0.996, 16: 0.949, 17: 1.068,
            18: 1.502, 19: 2.069, 20: 2.066, 21: 2.182, 22: 1.667, 23: 1.081
        }
        
        # Коэффициенты дней недели из EDA
        self.day_coefficients = {0: 1.094, 1: 0.956, 2: 1.209, 3: 1.044, 4: 0.938, 5: 1.290, 6: 1.580}
        
    def create_safe_features(self, hour, day_of_week, month, target_date):
        """Создает БЕЗОПАСНЫЕ признаки без утечек данных"""
        
        features = {}
        
        # 1. ЦИКЛИЧЕСКИЕ ПРИЗНАКИ (безопасно)
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # 2. БЕЗОПАСНЫЕ временные паттерны
        features['is_early_morning'] = 1 if 4 <= hour <= 6 else 0
        features['is_midday'] = 1 if 10 <= hour <= 16 else 0
        features['is_night'] = 1 if 0 <= hour <= 5 else 0
        
        # 3. СЕЗОННЫЕ ПАТТЕРНЫ
        features['is_high_season'] = 1 if month in [1, 2, 12] else 0
        features['is_low_season'] = 1 if month in [6, 7, 8] else 0
        features['is_spring'] = 1 if month in [3, 4, 5] else 0
        
        # 4. НЕДЕЛЬНЫЕ ПАТТЕРНЫ
        features['is_weekend'] = 1 if day_of_week >= 5 else 0
        features['is_monday'] = 1 if day_of_week == 0 else 0
        features['is_friday'] = 1 if day_of_week == 4 else 0
        
        # 5. КРИТИЧЕСКИЕ ПЕРИОДЫ (без утечек)
        features['morning_surge_6_7'] = 1 if 6 <= hour <= 7 else 0
        features['evening_surge_17_18'] = 1 if 17 <= hour <= 18 else 0
        
        # 6. БЕЗОПАСНЫЕ "лаги" на основе паттернов (не реальных данных!)
        base_value = self.hourly_avg[hour]
        day_coef = self.day_coefficients[day_of_week] / 1.156  # Нормализация
        
        features['lag_1h'] = base_value * day_coef * 0.95
        features['lag_24h'] = base_value * day_coef * 0.98
        
        # 7. СТАТИСТИКИ на основе паттернов
        all_values = list(self.hourly_avg.values())
        base_mean = np.mean(all_values)
        
        features['rolling_mean_6h'] = base_mean * day_coef
        features['rolling_mean_24h'] = base_mean * day_coef
        features['rolling_std_6h'] = np.std(all_values) * day_coef
        
        # 8. БАЗОВЫЕ ПРИЗНАКИ
        features['hour'] = hour
        features['day_of_week'] = day_of_week
        features['month'] = month
        
        return self._order_features(features)
    
    def _order_features(self, features):
        """Упорядочивает признаки как ожидает модель"""
        try:
            with open('models/feature_names.json', 'r', encoding='utf-8') as f:
                feature_names = json.load(f)
            
            # Заполняем недостающие признаки нулями
            ordered = {name: features.get(name, 0.0) for name in feature_names}
            return pd.DataFrame([ordered])
        except:
            # Если файл не найден, возвращаем как есть
            return pd.DataFrame([features])