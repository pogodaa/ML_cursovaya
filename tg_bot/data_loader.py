import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RealisticDataGenerator:
    def __init__(self):
        # Загружаем ТОЛЬКО для анализа паттернов, НЕ для конкретных прогнозов
        self.df = None
        self.load_pattern_data()
    
    def load_pattern_data(self):
        """Загружает данные ТОЛЬКО для анализа общих паттернов"""
        try:
            self.df = pd.read_csv('df/obr.csv', parse_dates=['datetime'], index_col='datetime')
            print(f"✅ Загружено {len(self.df)} записей для анализа паттернов")
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            self.df = None
    
    def get_general_patterns(self):
        """Возвращает ОБЩИЕ паттерны без привязки к конкретным датам"""
        if self.df is None:
            return self.get_fallback_patterns()
        
        # Анализируем ОБЩИЕ закономерности
        hourly_avg = self.df.groupby('hour')['Global_active_power'].mean().to_dict()
        monthly_factors = self.df.groupby('month')['Global_active_power'].mean() / self.df['Global_active_power'].mean()
        weekend_factor = self.df[self.df['is_weekend'] == 1]['Global_active_power'].mean() / self.df[self.df['is_weekend'] == 0]['Global_active_power'].mean()
        
        return {
            'hourly_avg': hourly_avg,
            'monthly_factors': monthly_factors.to_dict(),
            'weekend_factor': weekend_factor,
            'overall_avg': self.df['Global_active_power'].mean()
        }
    
    def get_fallback_patterns(self):
        """Резервные паттерны на основе вашего EDA"""
        return {
            'hourly_avg': REAL_HOURLY_AVERAGES,
            'monthly_factors': {1: 1.33, 2: 1.21, 3: 1.14, 4: 0.75, 5: 0.85, 6: 0.72},
            'weekend_factor': 1.37,
            'overall_avg': 1.156
        }

# Глобальные константы из ВАШЕГО EDA анализа
REAL_HOURLY_AVERAGES = {
    0: 0.778, 1: 0.634, 2: 0.540, 3: 0.517, 4: 0.489, 5: 0.527,
    6: 0.940, 7: 1.518, 8: 1.492, 9: 1.340, 10: 1.200, 11: 1.102,
    12: 1.054, 13: 1.000, 14: 1.040, 15: 0.996, 16: 0.949, 17: 1.068,
    18: 1.502, 19: 2.069, 20: 2.066, 21: 2.182, 22: 1.667, 23: 1.081
}