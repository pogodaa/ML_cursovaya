"""
ПАТТЕРНЫ ЭНЕРГОПОТРЕБЛЕНИЯ ИЗ EDA АНАЛИЗА
Автоматически сгенерировано из анализа данных
"""

# ==================== СУТОЧНЫЕ ПАТТЕРНЫ ====================
HOURLY_PATTERNS = {
    'mean': {
        0: 0.778, 1: 0.634, 2: 0.540, 3: 0.517, 4: 0.489, 5: 0.527,
        6: 0.940, 7: 1.518, 8: 1.492, 9: 1.340, 10: 1.200, 11: 1.102,
        12: 1.054, 13: 1.000, 14: 1.040, 15: 0.996, 16: 0.949, 17: 1.068,
        18: 1.502, 19: 2.069, 20: 2.066, 21: 2.182, 22: 1.667, 23: 1.081
    },
    'std': {
        0: 0.935, 1: 0.769, 2: 0.681, 3: 0.616, 4: 0.588, 5: 0.634,
        6: 1.092, 7: 1.137, 8: 1.057, 9: 0.948, 10: 0.994, 11: 1.000,
        12: 1.098, 13: 1.056, 14: 1.045, 15: 1.060, 16: 0.967, 17: 1.067,
        18: 1.331, 19: 1.600, 20: 1.544, 21: 1.466, 22: 1.236, 23: 1.009
    }
}

# Ключевые точки суточного цикла
HOURLY_KEY_POINTS = {
    'min_hour': 4,      # Минимум в 4:00
    'max_hour': 21,     # Максимум в 21:00  
    'min_value': 0.489,
    'max_value': 2.182,
    'range': 1.692,
    
    # Пиковые периоды
    'morning_peak_hours': [7, 8, 9],
    'evening_peak_hours': [18, 19, 20, 21, 22], 
    'night_hours': [0, 1, 2, 3, 4, 5],
    
    # Средние по периодам
    'morning_peak_avg': 1.450,
    'evening_peak_avg': 1.897,
    'night_avg': 0.581,
    
    # Критические переходы
    'biggest_increase': {'from': 6, 'to': 7, 'value': 0.579},
    'evening_increase': {'from': 17, 'to': 18, 'value': 0.434},
    'evening_decrease': {'from': 22, 'to': 23, 'value': -0.586}
}

# ==================== НЕДЕЛЬНЫЕ ПАТТЕРНЫ ====================
WEEKLY_PATTERNS = {
    'mean': {
        0: 1.094,  # Понедельник
        1: 0.956,  # Вторник
        2: 1.209,  # Среда  
        3: 1.044,  # Четверг
        4: 0.938,  # Пятница
        5: 1.290,  # Суббота
        6: 1.580   # Воскресенье
    },
    'median': {
        0: 0.574, 1: 0.436, 2: 0.842, 3: 0.414, 
        4: 0.392, 5: 0.708, 6: 1.322
    },
    'std': {
        0: 1.063, 1: 1.025, 2: 1.155, 3: 1.109,
        4: 0.984, 5: 1.288, 6: 1.425
    }
}

WEEKLY_KEY_POINTS = {
    'min_day': 4,           # Пятница (индекс 4)
    'max_day': 6,           # Воскресенье (индекс 6) 
    'min_value': 0.938,
    'max_value': 1.580,
    'range': 0.642,
    
    # Группы дней
    'workdays_avg': 1.048,      # Пн-Пт
    'weekends_avg': 1.432,      # Сб-Вс
    'week_start_avg': 1.086,    # Пн-Ср
    'week_end_avg': 0.991,      # Чт-Пт
    
    # Разницы
    'weekend_boost': 0.384,     # +36.6%
    'week_trend': -0.095        # -8.8%
}

# ==================== МЕСЯЧНЫЕ ПАТТЕРНЫ ====================
MONTHLY_PATTERNS = {
    'mean': {
        1: 1.546,  # Январь
        2: 1.401,  # Февраль
        3: 1.319,  # Март
        4: 0.863,  # Апрель
        5: 0.986,  # Май
        6: 0.827   # Июнь
    },
    'median': {
        1: 1.376, 2: 1.266, 3: 0.852, 
        4: 0.406, 5: 0.462, 6: 0.356
    },
    'std': {
        1: 1.292, 2: 1.312, 3: 1.276,
        4: 0.950, 5: 1.006, 6: 0.953
    }
}

MONTHLY_KEY_POINTS = {
    'min_month': 6,        # Июнь
    'max_month': 1,        # Январь
    'min_value': 0.827,
    'max_value': 1.546,
    'range': 0.719,
    
    # Сезоны
    'winter_avg': 1.474,   # Январь-Февраль
    'spring_avg': 1.056,   # Март-Май  
    'summer_avg': 0.827,   # Июнь
    'winter_to_summer_change': -0.647,  # -43.9%
    'winter_to_spring_change': -0.418,  # -28.3%
    
    # Тренд
    'monthly_trend': -0.120,  # кВт/месяц
    'total_change': -0.719    # за весь период
}

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def get_hourly_pattern(hour, pattern_type='mean'):
    """Возвращает паттерн потребления для указанного часа"""
    return HOURLY_PATTERNS[pattern_type].get(hour % 24, 1.0)

def get_daily_pattern(day_of_week, pattern_type='mean'):
    """Возвращает паттерн потребления для указанного дня недели"""
    return WEEKLY_PATTERNS[pattern_type].get(day_of_week % 7, 1.0)

def get_monthly_pattern(month, pattern_type='mean'):
    """Возвращает паттерн потребления для указанного месяца"""
    return MONTHLY_PATTERNS[pattern_type].get(month, 1.0)

def is_peak_hour(hour):
    """Проверяет, является ли час пиковым"""
    return hour in HOURLY_KEY_POINTS['morning_peak_hours'] or \
           hour in HOURLY_KEY_POINTS['evening_peak_hours']

def is_night_hour(hour):
    """Проверяет, является ли час ночным"""
    return hour in HOURLY_KEY_POINTS['night_hours']

def get_weekday_type(day_of_week):
    """Возвращает тип дня: workday/weekend"""
    return 'weekend' if day_of_week >= 5 else 'workday'