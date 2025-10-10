# bot.py - ВЕРСИЯ С КНОПКАМИ ДЛЯ СРАВНЕНИЯ
import telebot
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Загружаем токен
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Загружаем модель и признаки - ИСПРАВЛЕННЫЕ ПУТИ
try:
    model = joblib.load('models/dynamic_model_lightgbm.pkl')
    with open('models/dynamic_feature_names.json', 'r', encoding='utf-8') as f:
        FEATURE_NAMES = json.load(f)
    print(f"✅ Динамическая модель загружена. Ожидает {len(FEATURE_NAMES)} признаков")
except Exception as e:
    print(f"❌ Ошибка загрузки чистой модели: {e}")
    exit(1)

# ⚡ РЕАЛЬНЫЕ ДАННЫЕ ИЗ ВАШЕГО EDA АНАЛИЗА ⚡
REAL_HOURLY_AVERAGES = {
    # Час: среднее потребление (кВт) из вашего EDA
    0: 0.778, 1: 0.634, 2: 0.540, 3: 0.517, 4: 0.489, 5: 0.527,
    6: 0.940, 7: 1.518, 8: 1.492, 9: 1.340, 10: 1.200, 11: 1.102,
    12: 1.054, 13: 1.000, 14: 1.040, 15: 0.996, 16: 0.949, 17: 1.068,
    18: 1.502, 19: 2.069, 20: 2.066, 21: 2.182, 22: 1.667, 23: 1.081
}

# СЕЗОННЫЕ КОЭФФИЦИЕНТЫ ИЗ EDA
SEASONAL_FACTORS = {
    'winter': 1.474 / 1.156,  # Зимнее потребление / среднее
    'spring': 1.056 / 1.156,  # Весеннее потребление / среднее  
    'summer': 0.827 / 1.156,  # Летнее потребление / среднее
    'autumn': 1.100 / 1.156   # Осеннее (оценка)
}

# ДНЕВНЫЕ КОЭФФИЦИЕНТЫ ИЗ EDA
WEEKEND_FACTOR = 1.366  # Выходные +36.6%

class RealisticDataGenerator:
    def __init__(self):
        self.historical_predictions = {}
        
    def get_seasonal_factor(self, month):
        """Возвращает сезонный коэффициент на основе реальных данных EDA"""
        if month in [12, 1, 2]:  # Зима
            return SEASONAL_FACTORS['winter']
        elif month in [3, 4, 5]:  # Весна
            return SEASONAL_FACTORS['spring'] 
        elif month in [6, 7, 8]:  # Лето
            return SEASONAL_FACTORS['summer']
        else:  # Осень
            return SEASONAL_FACTORS['autumn']
    
    def get_day_factor(self, day_of_week):
        """Коэффициент дня недели из EDA"""
        return WEEKEND_FACTOR if day_of_week >= 5 else 1.0

    def generate_realistic_base_consumption(self, hour, day_of_week, month):
        """Генерирует базовое потребление БЛИЗКОЕ К ОБУЧАЮЩИМ ДАННЫМ"""
        # Вместо реальных средних из EDA используем значения ближе к train данным
        base = 1.2  # Среднее train данных
        
        # Небольшие корректировки по часам (но не такие сильные как в EDA)
        hour_adjustment = 1.0
        if 0 <= hour <= 5:   # Ночь
            hour_adjustment = 0.7
        elif 7 <= hour <= 9:  # Утро
            hour_adjustment = 1.3  
        elif 18 <= hour <= 22:  # Вечер
            hour_adjustment = 1.4
        
        consumption = base * hour_adjustment * np.random.uniform(0.9, 1.1)
        return max(0.5, min(3.0, consumption))

    # def get_realistic_lags(self, current_hour, current_consumption, target_date):
    #     """Генерирует реалистичные лаги КАК В ОБУЧАЮЩИХ ДАННЫХ"""
    #     # Используем значения близкие к обучающим данным (1.0-1.5 кВт)
    #     base_value = 1.2  # Близко к среднему train данных
        
    #     # Лаги с вариациями как в реальных данных
    #     lag_24h = base_value * np.random.uniform(0.8, 1.2)
    #     lag_48h = base_value * np.random.uniform(0.7, 1.3)
    #     lag_72h = base_value * np.random.uniform(0.6, 1.4)
    #     lag_96h = base_value * np.random.uniform(0.5, 1.5)
    #     lag_168h = base_value * np.random.uniform(0.4, 1.6)
        
    #     # Взаимодействия
    #     is_morning = 1 if 7 <= current_hour <= 9 else 0
    #     is_evening = 1 if 18 <= current_hour <= 22 else 0
    #     is_night = 1 if 0 <= current_hour <= 5 else 0
    #     is_weekend = 1 if target_date.weekday() >= 5 else 0
    #     is_winter = 1 if target_date.month in [12, 1, 2] else 0
        
    #     return {
    #         'lag_same_day_24h': max(0.1, lag_24h),
    #         'lag_week_ago_168h': max(0.1, lag_168h),
    #         'lag_48h_ago': max(0.1, lag_48h),
    #         'lag_72h_ago': max(0.1, lag_72h),
    #         'lag_96h_ago': max(0.1, lag_96h),
            
    #         # Взаимодействия
    #         'lag_24h_morning': max(0.1, lag_24h * is_morning),
    #         'lag_24h_evening': max(0.1, lag_24h * is_evening),
    #         'lag_24h_night': max(0.1, lag_24h * is_night),
    #         'lag_24h_weekend': max(0.1, lag_24h * is_weekend),
    #         'lag_48h_morning': max(0.1, lag_48h * is_morning),
    #         'lag_48h_evening': max(0.1, lag_48h * is_evening),
    #         'lag_week_morning': max(0.1, lag_168h * is_morning),
    #         'lag_week_evening': max(0.1, lag_168h * is_evening),
    #         'lag_24h_winter': max(0.1, lag_24h * is_winter),
    #         'lag_168h_weekend': max(0.1, lag_168h * is_weekend)
    #     }

    def get_realistic_lags(self, current_hour, day_of_week, month, target_date):
        """Реалистичные лаги на основе дня недели и сезона"""
        
        # Базовое потребление по паттернам EDA
        base_weekday = 1.2  # Будни
        base_weekend = 1.5  # Выходные (выше на 25%)
        
        # Сезонные корректировки
        seasonal_factor = 1.0
        if month in [12, 1, 2]:  # Зима
            seasonal_factor = 1.3
        elif month in [6, 7, 8]:  # Лето  
            seasonal_factor = 0.8
        
        # Выбираем базу по дню недели
        base = base_weekend if day_of_week >= 5 else base_weekday
        base *= seasonal_factor
        
        # Лаги с вариациями + тренды
        lag_24h = base * np.random.uniform(0.9, 1.1)  # Вчера
        lag_48h = base * np.random.uniform(0.85, 1.15)  # Позавчера
        lag_168h = base * np.random.uniform(0.8, 1.2)   # Неделю назад
        
        # Выходные имеют другой паттерн!
        if day_of_week >= 5:  # Выходные
            lag_24h *= 1.1  # Вчера было воскресенье - потребление выше
            lag_168h *= 0.9  # Неделю назад был будний день
        else:  # Будни
            lag_24h *= 0.9  # Вчера был понедельник - потребление ниже
            lag_168h *= 1.1  # Неделю назад были выходные
        
        return {
            'lag_same_day_24h': max(0.3, lag_24h),
            'lag_week_ago_168h': max(0.3, lag_168h),
            'lag_48h_ago': max(0.3, lag_48h),
            'lag_72h_ago': max(0.3, base * np.random.uniform(0.8, 1.2)),
            'lag_96h_ago': max(0.3, base * np.random.uniform(0.75, 1.25)),
        }

    def get_realistic_rolling_stats(self, hour, month):
        """Реалистичные скользящие статистики на основе EDA"""
        # 24-часовое среднее - из общих статистик EDA
        all_hourly_values = list(REAL_HOURLY_AVERAGES.values())
        rolling_24h_mean = np.mean(all_hourly_values) * self.get_seasonal_factor(month)
        
        # 7-дневное среднее (168 часов)
        rolling_168h = rolling_24h_mean * np.random.uniform(0.99, 1.01)
        
        # 7-дневное среднее (альтернативное название)
        rolling_7d_past = rolling_24h_mean * np.random.uniform(0.98, 1.02)
        
        return {
            'rolling_mean_24h': max(0.1, rolling_24h_mean),
            'rolling_mean_168h': max(0.1, rolling_168h),
            'rolling_mean_7d_past': max(0.1, rolling_7d_past)
        }
    
    def get_realistic_submetering(self, hour, day_of_week, month):
        """Реалистичные данные суб-счетчиков на основе анализа EDA"""
        # Активность основана на реальных паттернах
        kitchen_active = 1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0
        laundry_active = 1 if (10 <= hour <= 18) and (day_of_week >= 5) else 0
        ac_heating_active = 1 if ((18 <= hour <= 22) and (month in [12, 1, 2])) or \
                                ((13 <= hour <= 17) and (month in [6, 7, 8])) else 0
        
        # Соотношения на основе анализа потребления
        if kitchen_active:
            kitchen_ratio = np.random.uniform(0.15, 0.25)
        else:
            kitchen_ratio = np.random.uniform(0.02, 0.08)
            
        if laundry_active:
            laundry_ratio = np.random.uniform(0.08, 0.15)
        else:
            laundry_ratio = np.random.uniform(0.01, 0.04)
            
        if ac_heating_active:
            ac_heating_ratio = np.random.uniform(0.25, 0.35)
        else:
            ac_heating_ratio = np.random.uniform(0.05, 0.12)
        
        return {
            'kitchen_ratio': kitchen_ratio,
            'laundry_ratio': laundry_ratio,
            'ac_heating_ratio': ac_heating_ratio,
            'kitchen_active': kitchen_active,
            'laundry_active': laundry_active,
            'ac_heating_active': ac_heating_active
        }
    
    def debug_feature_generation(hour, day_of_week, month, target_date):
        """Отладочная информация о генерируемых признаках"""
        features_df = create_realistic_features(hour, day_of_week, month, target_date)
        
        print(f"\n=== ОТЛАДКА ЧАС {hour} ===")
        print(f"Дата: {target_date}, День недели: {day_of_week}, Месяц: {month}")
        
        # Проверь ключевые признаки
        key_features = ['lag_same_day_24h', 'rolling_mean_24h', 'hour_sin', 'hour_cos', 
                    'is_evening_peak', 'is_morning_peak', 'is_night']
        
        for feature in key_features:
            if feature in features_df.columns:
                print(f"  {feature}: {features_df[feature].iloc[0]:.3f}")
        
        # Предсказание
        prediction = model.predict(features_df)[0]
        print(f"  ПРОГНОЗ: {prediction:.3f} кВт")
        
        return prediction

# Инициализация генератора
data_gen = RealisticDataGenerator()

def create_realistic_features(hour, day_of_week, month, target_date):
    """Создает РЕАЛИСТИЧНЫЕ признаки на основе реальных данных EDA"""
    
    features = {}
    
    # 1. ЦИКЛИЧЕСКИЕ ПРИЗНАКИ
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # 2. СУТОЧНЫЕ ПАТТЕРНЫ ИЗ EDA
    features['is_evening_peak'] = 1 if 18 <= hour <= 22 else 0
    features['is_morning_peak'] = 1 if 7 <= hour <= 9 else 0
    features['is_night'] = 1 if 0 <= hour <= 5 else 0
    features['is_midday'] = 1 if 10 <= hour <= 16 else 0
    features['is_early_morning'] = 1 if 4 <= hour <= 6 else 0
    features['is_late_evening'] = 1 if 21 <= hour <= 23 else 0
    features['is_deep_night'] = 1 if 1 <= hour <= 4 else 0
    
    # 3. НЕДЕЛЬНЫЕ ПАТТЕРНЫ ИЗ EDA
    features['is_monday'] = 1 if day_of_week == 0 else 0
    features['is_friday'] = 1 if day_of_week == 4 else 0
    features['is_sunday'] = 1 if day_of_week == 6 else 0
    features['is_week_start'] = 1 if day_of_week in [0, 1] else 0
    features['is_week_end'] = 1 if day_of_week in [4, 5] else 0
    features['weekend_evening_boost'] = 1 if (day_of_week >= 5 and 18 <= hour <= 22) else 0
    features['weekend_morning'] = 1 if (day_of_week >= 5 and 7 <= hour <= 9) else 0
    
    # 4. СЕЗОННЫЕ ПАТТЕРНЫ ИЗ EDA
    features['is_high_season'] = 1 if month in [12, 1, 2] else 0
    features['is_low_season'] = 1 if month in [6, 7, 8] else 0
    features['is_spring'] = 1 if month in [3, 4, 5] else 0
    
    # 5. КРИТИЧЕСКИЕ ПЕРЕХОДЫ ИЗ EDA
    features['morning_surge_6_7'] = 1 if 6 <= hour <= 7 else 0
    features['evening_surge_17_18'] = 1 if 17 <= hour <= 18 else 0
    features['evening_drop_22_23'] = 1 if 22 <= hour <= 23 else 0
    
    # 6. ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ
    features['winter_evening'] = 1 if (month in [12, 1, 2] and 18 <= hour <= 22) else 0
    features['summer_afternoon'] = 1 if (month in [6, 7, 8] and 10 <= hour <= 16) else 0
    features['workday_evening'] = 1 if (day_of_week < 5 and 18 <= hour <= 22) else 0
    features['sunday_evening'] = 1 if (day_of_week == 6 and 18 <= hour <= 22) else 0
    
    # 7. БАЗОВЫЕ ПРИЗНАКИ
    features['hour'] = hour
    features['day_of_week'] = day_of_week
    features['month'] = month
    features['is_weekend'] = 1 if day_of_week >= 5 else 0

    # 8. СУБ-СЧЕТЧИКИ - ЗАКОММЕНТИРОВАТЬ или ЗАПОЛНИТЬ НУЛЯМИ
    # features['Sub_metering_1'] = 0.1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0.0
    # features['Sub_metering_2'] = 0.05 if (10 <= hour <= 18) and (day_of_week >= 5) else 0.0  
    # features['Sub_metering_3'] = 0.2 if ((18 <= hour <= 22) and (month in [12, 1, 2])) or ((13 <= hour <= 17) and (month in [6, 7, 8])) else 0.0
    
    # ЗАПОЛНИ ИХ НУЛЯМИ:
    features['Sub_metering_1'] = 0.0
    features['Sub_metering_2'] = 0.0
    features['Sub_metering_3'] = 0.0
    features['kitchen_active'] = 0
    features['laundry_active'] = 0  
    features['ac_heating_active'] = 0

    # 9. ДОБАВЛЯЕМ НЕДОСТАЮЩИЕ ПРИЗНАКИ ИЗ safe_feature_names.json
    # Просто заполняем их нулями или базовыми значениями
    all_safe_features = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_evening_peak', 'is_morning_peak', 
        'is_night', 'is_midday', 'is_early_morning', 'is_late_evening', 'is_deep_night',
        'is_monday', 'is_friday', 'is_sunday', 'is_week_start', 'is_week_end',
        'weekend_evening_boost', 'weekend_morning', 'is_high_season', 'is_low_season',
        'is_spring', 'morning_surge_6_7', 'evening_surge_17_18', 'evening_drop_22_23',
        'winter_evening', 'summer_afternoon', 'workday_evening', 'sunday_evening',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'kitchen_active', 
        'laundry_active', 'ac_heating_active'
    ]
    
    # Заполняем все признаки
    for feature in all_safe_features:
        if feature not in features:
            if 'active' in feature:
                features[feature] = 0  # для бинарных
            else:
                features[feature] = 0.0  # для числовых
    
    # Создаем DataFrame в правильном порядке
    ordered_features = {name: features[name] for name in FEATURE_NAMES}
    
    return pd.DataFrame([ordered_features])


def predict_for_date(target_date):
    """Прогноз для конкретной даты"""
    day_of_week = target_date.weekday()
    month = target_date.month
    date_str = target_date.strftime('%Y-%m-%d')
    
    predictions = []
    
    print(f"📅 Прогноз на {target_date.strftime('%d.%m.%Y')} ({['пн','вт','ср','чт','пт','сб','вс'][day_of_week]}, месяц {month})")
    
    # Прогнозируем каждый час
    for hour in range(24):
        # Создаем РЕАЛИСТИЧНЫЕ признаки для этого часа
        features_df = create_realistic_features(hour, day_of_week, month, target_date)
        
        # Прогнозируем
        prediction = model.predict(features_df)[0]
        predictions.append(max(0.1, min(5.0, prediction)))
        
        print(f"  Час {hour:2d}: {predictions[-1]:.2f} кВт")
    
    # Сохраняем для использования в будущих лагах
    data_gen.historical_predictions[date_str] = {
        hour: pred for hour, pred in enumerate(predictions)
    }
    
    return list(range(24)), predictions, day_of_week, month

def create_comparison_plot(hours, predictions_tomorrow, predictions_day_after, date_tomorrow, date_day_after):
    """Создает график сравнения двух прогнозов с ночным пиком"""
    plt.figure(figsize=(14, 8))
    
    # Графики прогнозов
    plt.plot(hours, predictions_tomorrow, 'b-', linewidth=3, marker='o', markersize=4, 
             label=f'Завтра ({date_tomorrow})', alpha=0.8)
    
    plt.plot(hours, predictions_day_after, 'r-', linewidth=3, marker='s', markersize=4, 
             label=f'Послезавтра ({date_day_after})', alpha=0.8)
    
    # Реальные средние значения из EDA для сравнения
    real_values = [REAL_HOURLY_AVERAGES[h] for h in hours]
    plt.plot(hours, real_values, 'g--', linewidth=2, label='Реальные средние из EDA', alpha=0.6)
    
    # Зоны пиков из EDA - ТЕПЕРЬ С НОЧНЫМ ПИКОМ!
    plt.axvspan(0, 5, alpha=0.15, color='blue', label='Ночное время (0-5)')
    plt.axvspan(7, 9, alpha=0.15, color='orange', label='Утренний пик (7-9)')
    plt.axvspan(18, 22, alpha=0.15, color='red', label='Вечерний пик (18-22)')
    
    plt.title('Сравнение прогнозов энергопотребления\n(Честная оценка работы модели)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Час дня', fontsize=12)
    plt.ylabel('Нагрузка (кВт)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, 24, 2))
    plt.ylim(bottom=0, top=5)
    
    # Добавляем аннотации для пиковых зон
    plt.text(2.5, 0.5, '', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='blue', alpha=0.7)
    plt.text(8, 0.5, '', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='orange', alpha=0.7)
    plt.text(20, 0.5, '', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='red', alpha=0.7)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_realistic_features(hour, day_of_week, month, target_date):
    """Создает ДИНАМИЧЕСКИЕ признаки для новой модели"""
    
    features = {}
    
    # 1. ЦИКЛИЧЕСКИЕ ПРИЗНАКИ
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # 2. БАЗОВЫЕ ВРЕМЕННЫЕ
    features['hour'] = hour
    features['day_of_week'] = day_of_week
    features['month'] = month
    features['is_weekend'] = 1 if day_of_week >= 5 else 0
    
    # 3. ПАТТЕРНЫ
    features['is_evening_peak'] = 1 if 18 <= hour <= 22 else 0
    features['is_morning_peak'] = 1 if 7 <= hour <= 9 else 0
    features['is_night'] = 1 if 0 <= hour <= 5 else 0
    
    # 4. КЛЮЧЕВЫЕ - ВЗАИМОДЕЙСТВИЯ ДНЯ И ВРЕМЕНИ
    features['weekday_morning'] = 1 if (day_of_week < 5 and 7 <= hour <= 9) else 0
    features['weekday_evening'] = 1 if (day_of_week < 5 and 18 <= hour <= 22) else 0
    features['weekend_morning'] = 1 if (day_of_week >= 5 and 7 <= hour <= 9) else 0
    features['weekend_evening'] = 1 if (day_of_week >= 5 and 18 <= hour <= 22) else 0
    
    # 5. СЕЗОННЫЕ ВЗАИМОДЕЙСТВИЯ
    features['winter_evening'] = 1 if (month in [12, 1, 2] and 18 <= hour <= 22) else 0
    features['summer_afternoon'] = 1 if (month in [6, 7, 8] and 13 <= hour <= 17) else 0
    
    # 6. ПОВЕДЕНЧЕСКИЕ ПАТТЕРНЫ
    features['family_time'] = 1 if (18 <= hour <= 22 and day_of_week < 5) else 0
    features['late_night_weekend'] = 1 if ((23 <= hour <= 23) or (0 <= hour <= 2)) and day_of_week >= 5 else 0
    
    # 7. СУБ-СЧЕТЧИКИ (заполняем нулями)
    features['Sub_metering_1'] = 0.0
    features['Sub_metering_2'] = 0.0
    features['Sub_metering_3'] = 0.0
    
    # Создаем DataFrame в правильном порядке
    ordered_features = {name: features[name] for name in FEATURE_NAMES}
    
    return pd.DataFrame([ordered_features])

def create_prediction_keyboard():
    """Создает клавиатуру с кнопками для прогнозов"""
    keyboard = InlineKeyboardMarkup()
    keyboard.row(
        InlineKeyboardButton("📅 Завтра", callback_data="predict_tomorrow"),
        InlineKeyboardButton("📆 Послезавтра", callback_data="predict_day_after")
    )
    keyboard.row(
        InlineKeyboardButton("📊 Сравнить оба", callback_data="compare_both")
    )
    return keyboard

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 *Бот прогнозирования энергопотребления*

*ЧЕСТНАЯ оценка работы ML модели*

*Важно:* Результаты показывают реальные проблемы модели:
• Ночное потребление завышено
• Утренние пики занижены  
• Это НОРМАЛЬНО для учебного проекта!

*Команды:*
/predict - Прогноз с сравнением
/stats - Статистика и анализ проблем

*Используйте кнопки ниже для тестирования:*
    """
    bot.send_message(message.chat.id, welcome_text, 
                   parse_mode='Markdown',
                   reply_markup=create_prediction_keyboard())

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    try:
        if call.data == "predict_tomorrow":
            bot.answer_callback_query(call.id, "Генерирую прогноз на завтра...")
            send_single_prediction(call.message, days_ahead=1)
            
        elif call.data == "predict_day_after":
            bot.answer_callback_query(call.id, "Генерирую прогноз на послезавтра...")
            send_single_prediction(call.message, days_ahead=2)
            
        elif call.data == "compare_both":
            bot.answer_callback_query(call.id, "Сравниваю оба прогноза...")
            send_comparison(call.message)
            
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Ошибка: {str(e)}")

def send_single_prediction(message, days_ahead=1):
    """Отправляет прогноз для одного дня"""
    try:
        target_date = datetime.now() + timedelta(days=days_ahead)
        hours, predictions, day_of_week, month = predict_for_date(target_date)
        
        date_str = target_date.strftime('%d.%m.%Y')
        day_names = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        
        # Создаем график
        plt.figure(figsize=(12, 6))
        plt.plot(hours, predictions, 'b-', linewidth=2, marker='o', label='Прогноз ML')
        plt.plot(hours, [REAL_HOURLY_AVERAGES[h] for h in hours], 'r--', label='Реальные средние')
        plt.title(f'Прогноз на {date_str} ({day_names[day_of_week]})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        # ДОБАВЬ после plt.legend():
        plt.ylim(0, 5.0)  # Установи одинаковые границы для всех графиков
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Статистика
        avg = np.mean(predictions)
        peak = np.max(predictions)
        peak_hour = hours[np.argmax(predictions)]
        
        caption = f"""📊 *Прогноз на {date_str}*
*{day_names[day_of_week]}*

*Метрики:*
• Средняя нагрузка: {avg:.2f} кВт
• Пиковая нагрузка: {peak:.2f} кВт в {peak_hour}:00

*Анализ:*
Ночное потребление: {predictions[0]:.2f} кВт (ожидалось ~0.78 кВт)
Утренний пик: {predictions[7]:.2f} кВт (ожидалось ~1.52 кВт)

*Вывод:* Модель имеет систематические ошибки"""
        
        bot.send_photo(message.chat.id, buf, caption=caption, parse_mode='Markdown',
                      reply_markup=create_prediction_keyboard())
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка: {str(e)}")

def send_comparison(message):
    """Отправляет сравнение двух прогнозов"""
    try:
        # Прогноз на завтра
        tomorrow = datetime.now() + timedelta(days=1)
        hours, pred_tomorrow, dow_tomorrow, month_tomorrow = predict_for_date(tomorrow)
        
        # Прогноз на послезавтра
        day_after = datetime.now() + timedelta(days=2)
        _, pred_day_after, dow_day_after, month_day_after = predict_for_date(day_after)
        
        # Создаем график сравнения
        plot_buf = create_comparison_plot(hours, pred_tomorrow, pred_day_after,
                                         tomorrow.strftime('%d.%m'), day_after.strftime('%d.%m'))
        
        # Анализ различий
        avg_tomorrow = np.mean(pred_tomorrow)
        avg_day_after = np.mean(pred_day_after)
        diff_avg = abs(avg_tomorrow - avg_day_after)
        
        # Находим максимальное различие по часам
        hour_diffs = [abs(p1 - p2) for p1, p2 in zip(pred_tomorrow, pred_day_after)]
        max_diff = max(hour_diffs)
        max_diff_hour = hours[np.argmax(hour_diffs)]
        
        caption = f"""📊 *Сравнение прогнозов*

*Статистика:*
• Завтра: {avg_tomorrow:.2f} кВт (среднее)
• Послезавтра: {avg_day_after:.2f} кВт (среднее)
• Разница: {diff_avg:.2f} кВт

*Максимальное различие:*
{max_diff:.2f} кВт в {max_diff_hour}:00

*Честная оценка модели:*
✅ Прогнозы РАЗНЫЕ для разных дней
⚠️ Но есть систематические ошибки
📈 Общий паттерн сохраняется

*Это реалистичный результат для учебного проекта!*"""
        
        bot.send_photo(message.chat.id, plot_buf, caption=caption, parse_mode='Markdown',
                      reply_markup=create_prediction_keyboard())
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка сравнения: {str(e)}")

@bot.message_handler(commands=['predict'])
def send_predict_menu(message):
    """Меню прогнозов"""
    menu_text = """
📊 *Тестирование модели прогнозирования*

Выберите опцию для проверки работы модели:

• *Завтра* - прогноз на 1 день вперед
• *Послезавтра* - прогноз на 2 дня вперед  
• *Сравнить оба* - анализ различий между днями

*Цель:* Убедиться что прогнозы РАЗНЫЕ для разных дат
и оценить реальное качество модели.
    """
    bot.send_message(message.chat.id, menu_text, 
                   parse_mode='Markdown',
                   reply_markup=create_prediction_keyboard())

@bot.message_handler(commands=['stats'])
def send_stats(message):
    stats_text = """
📊 *Честная статистика модели*

*Результаты тестирования:*
⚠️ *Проблемы выявленные в работе:*
• Ночное потребление завышено в 2-3 раза
• Утренние пики занижены на 20-30%
• Общий уровень потребления выше реального

*Это НОРМАЛЬНО потому что:*
1. Модель переобучилась на обучающих данных
2. Недостаточно данных для генерализации
3. Проблема переноса на новые даты

*Технические метрики:*
• LightGBM R²: 87.95% (на тестовых данных)
• Средняя ошибка: 0.18 кВт
• Но на новых данных ошибка больше!

*Вывод:* Модель работает, но требует доработки.
Отличный результат для учебного проекта! 🎯
    """
    bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    help_text = """
🤖 Я бот для ЧЕСТНОЙ оценки ML модели.

Используйте /predict для тестирования модели
или кнопки ниже для быстрого доступа.

⚠️ Помните: мы тестируем РЕАЛЬНУЮ работу модели,
а не пытаемся получить идеальные результаты!
    """
    bot.send_message(message.chat.id, help_text,
                   reply_markup=create_prediction_keyboard())
    

@bot.message_handler(commands=['debug'])
def send_debug(message):
    """Диагностика проблемы с моделью"""
    try:
        debug_result = diagnose_bot_problem()
        bot.send_message(message.chat.id, debug_result, parse_mode='Markdown')
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка диагностики: {str(e)}")

def diagnose_bot_problem():
    """Диагностика проблемы в боте - ВЕРНУТЬ РЕЗУЛЬТАТ ДЛЯ ОТПРАВКИ"""
    result = "🔍 *ДИАГНОСТИКА МОДЕЛИ И БОТА*\n\n"
    
    # Тест на конкретную дату и час
    test_date = datetime.now() + timedelta(days=1)
    hour = 3  # Ночное время
    day_of_week = test_date.weekday()
    month = test_date.month
    
    # 1. Создаем признаки
    features_df = create_realistic_features(hour, day_of_week, month, test_date)
    
    result += "1. *КЛЮЧЕВЫЕ ПРИЗНАКИ:*\n"
    key_features = [
        'rolling_mean_24h', 'lag_same_day_24h', 'lag_week_ago_168h',
        'hour_sin', 'hour_cos', 'is_night', 'is_evening_peak'
    ]
    
    for feature in key_features:
        if feature in features_df.columns:
            result += f"   `{feature}: {features_df[feature].iloc[0]:.3f}`\n"
    
    # 2. Делаем предсказание
    prediction = model.predict(features_df)[0]
    result += f"\n2. *ПРЕДСКАЗАНИЕ:* `{prediction:.3f} кВт`\n"
    
    # 3. Анализ проблемы
    result += f"\n3. *АНАЛИЗ:*\n"
    if prediction > 2.0:
        result += "   ❌ *ПРОБЛЕМА:* Предсказание слишком высокое\n"
        result += "   *Возможные причины:*\n"
        result += "   - Модель переобучилась\n"
        result += "   - Признаки не совпадают с обучением\n"
        result += "   - Есть утечки данных\n"
    else:
        result += "   ✅ Предсказание в норме\n"
    
    # 4. Тест нескольких часов
    result += f"\n4. *ТЕСТ РАЗНЫХ ЧАСОВ:*\n"
    test_hours = [0, 3, 7, 12, 18, 22]
    for h in test_hours:
        features = create_realistic_features(h, day_of_week, month, test_date)
        pred = model.predict(features)[0]
        result += f"   Час {h:2d}: `{pred:.3f} кВт`\n"
    
    return result


@bot.message_handler(commands=['test'])
def quick_test(message):
    """Быстрый тест модели"""
    try:
        # Простой тест - создаем минимальные признаки
        test_date = datetime.now()
        hour = test_date.hour
        day_of_week = test_date.weekday()
        month = test_date.month
        
        features_df = create_realistic_features(hour, day_of_week, month, test_date)
        prediction = model.predict(features_df)[0]
        
        response = f"""
⚡ *БЫСТРЫЙ ТЕСТ МОДЕЛИ*

*Текущее время:* {test_date.strftime('%H:%M')}
*Предсказание:* `{prediction:.3f} кВт`

*Ожидалось:* 0.8-1.5 кВт
*Результат:* {'❌ ВЫСОКОЕ' if prediction > 2.0 else '✅ НОРМА'}

*Вывод:* {'Модель предсказывает завышенные значения' if prediction > 2.0 else 'Модель работает нормально'}
        """
        
        bot.send_message(message.chat.id, response, parse_mode='Markdown')
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка теста: {str(e)}")


def console_diagnostic():
    """Диагностика в консоли при запуске"""
    print("\n" + "="*60)
    print("🚨 ДИАГНОСТИКА ПРИ ЗАПУСКЕ")
    print("="*60)
    
    # Тест 1: Проверка загрузки модели
    print("1. ПРОВЕРКА МОДЕЛИ:")
    print(f"   ✅ Модель загружена: {type(model)}")
    print(f"   ✅ Признаков ожидается: {len(FEATURE_NAMES)}")
    
    # Тест 2: Проверка предсказания на тестовых данных
    print("2. ТЕСТ ПРЕДСКАЗАНИЙ:")
    
    test_cases = [
        (3, 0, 10, "Ночь будний"),   # час, день_недели, месяц, описание
        (8, 0, 10, "Утро будний"),
        (19, 0, 10, "Вечер будний"),
        (14, 5, 10, "День выходной")
    ]
    
    for hour, day, month, desc in test_cases:
        test_date = datetime(2025, month, 15)  # Фиксированная дата
        features = create_realistic_features(hour, day, month, test_date)
        prediction = model.predict(features)[0]
        
        status = "❌ ВЫСОКО" if prediction > 2.0 else "✅ НОРМА"
        print(f"   {desc}: {prediction:.3f} кВт - {status}")
    
    print("3. ВЫВОД:")
    print("   🎯 Запускаю бота для дальнейшего тестирования...")
    print("="*60)



# Добавь ПЕРЕД bot.infinity_polling():
def test_dynamic_features():
    """Тест динамических признаков"""
    print("\n=== ТЕСТ ДИНАМИЧЕСКИХ ПРИЗНАКОВ ===")
    
    # Тест разных дней
    test_cases = [
        (8, 0, 10, "Понедельник утро"),
        (8, 5, 10, "Суббота утро"), 
        (20, 0, 10, "Понедельник вечер"),
        (20, 6, 10, "Воскресенье вечер")
    ]
    
    for hour, day, month, desc in test_cases:
        test_date = datetime(2025, month, 15)
        features_df = create_realistic_features(hour, day, month, test_date)
        
        # Проверяем ключевые признаки
        weekday_morning = features_df['weekday_morning'].iloc[0]
        weekend_evening = features_df['weekend_evening'].iloc[0]
        
        prediction = model.predict(features_df)[0]
        
        print(f"  {desc}: {prediction:.3f} кВт")
        print(f"     weekday_morning: {weekday_morning}, weekend_evening: {weekend_evening}")



if __name__ == "__main__":
    print("🚀 Бот запущен для ЧЕСТНОЙ оценки модели!")
    print("📊 Кнопки для сравнения прогнозов активированы")
    print("⚠️  Ожидаем выявления реальных проблем модели")
    # Вызови тест при запуске
    test_dynamic_features()
    bot.infinity_polling()
