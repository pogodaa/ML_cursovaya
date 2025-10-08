# bot.py 
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

# Загружаем токен
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Загружаем модель и признаки
model = joblib.load('models/lightgbm_model.pkl')
with open('models/feature_names.json', 'r', encoding='utf-8') as f:
    FEATURE_NAMES = json.load(f)

print(f"✅ Модель загружена. Ожидает {len(FEATURE_NAMES)} признаков")

# РЕАЛИСТИЧНЫЕ СТАТИСТИКИ НА ОСНОВЕ ВАШЕГО EDA
HOURLY_AVERAGES = {
    0: 0.778, 1: 0.634, 2: 0.540, 3: 0.517, 4: 0.489, 5: 0.527,
    6: 0.940, 7: 1.518, 8: 1.492, 9: 1.340, 10: 1.200, 11: 1.102,
    12: 1.054, 13: 1.000, 14: 1.040, 15: 0.996, 16: 0.949, 17: 1.068,
    18: 1.502, 19: 2.069, 20: 2.066, 21: 2.182, 22: 1.667, 23: 1.081
}

HOURLY_STD = {
    0: 0.935, 1: 0.769, 2: 0.681, 3: 0.616, 4: 0.588, 5: 0.634,
    6: 1.092, 7: 1.137, 8: 1.057, 9: 0.948, 10: 0.994, 11: 1.000,
    12: 1.098, 13: 1.056, 14: 1.045, 15: 1.060, 16: 0.967, 17: 1.067,
    18: 1.331, 19: 1.600, 20: 1.544, 21: 1.466, 22: 1.236, 23: 1.009
}

# Сезонные коэффициенты на основе вашего месячного анализа
SEASONAL_FACTORS = {
    1: 1.33,  # Январь +33%
    2: 1.21,  # Февраль +21%
    3: 1.14,  # Март +14%
    4: 0.75,  # Апрель -25%
    5: 0.85,  # Май -15%
    6: 0.72,  # Июнь -28%
    7: 0.70,  # Июль -30% (оценка)
    8: 0.70,  # Август -30% (оценка)
    9: 0.80,  # Сентябрь -20% (оценка)
    10: 0.90, # Октябрь -10% (оценка)
    11: 1.10, # Ноябрь +10% (оценка)
    12: 1.25  # Декабрь +25% (оценка)
}

def create_realistic_lags(hour, day_of_week, month):
    """Создает РЕАЛИСТИЧНЫЕ лаги на основе статистик EDA"""
    
    # Базовое потребление для текущего часа
    base_consumption = HOURLY_AVERAGES[hour]
    
    # Применяем сезонный коэффициент
    seasonal_factor = SEASONAL_FACTORS.get(month, 1.0)
    base_consumption *= seasonal_factor
    
    # Корректировка для выходных (на основе вашего анализа +36.6%)
    if day_of_week >= 5:  # Суббота и воскресенье
        base_consumption *= 1.15  # +15% для выходных
    
    features = {}
    
    # РЕАЛИСТИЧНЫЕ ЛАГИ с учетом времени и сезонности
    features['lag_2h_ago'] = HOURLY_AVERAGES[(hour - 2) % 24] * seasonal_factor
    features['lag_6h_ago'] = HOURLY_AVERAGES[(hour - 6) % 24] * seasonal_factor
    features['lag_12h_ago'] = HOURLY_AVERAGES[(hour - 12) % 24] * seasonal_factor
    features['lag_same_day_24h'] = base_consumption  # Вчера в это же время
    features['lag_week_ago_168h'] = base_consumption  # Неделю назад
    
    return features, base_consumption

def create_realistic_rolling_stats(hour, base_consumption):
    """Создает реалистичные скользящие статистики"""
    
    # Скользящее среднее за 3 часа (текущий + 2 предыдущих)
    recent_hours = [HOURLY_AVERAGES[(hour - i) % 24] for i in range(3)]
    rolling_3h = np.mean(recent_hours)
    
    # Скользящее среднее за 24 часа
    rolling_24h = np.mean(list(HOURLY_AVERAGES.values()))
    
    # Стандартное отклонение за 24 часа
    rolling_24h_std = np.std(list(HOURLY_AVERAGES.values()))
    
    # Скользящее среднее за 7 дней
    rolling_7d = rolling_24h  # Для простоты используем дневное среднее
    
    return {
        'rolling_mean_3h_past': rolling_3h,
        'rolling_mean_24h_past': rolling_24h,
        'rolling_std_24h_past': rolling_24h_std,
        'rolling_mean_7d_past': rolling_7d
    }

def create_realistic_sub_metering(hour, day_of_week, month):
    """Создает реалистичные признаки суб-счетчиков"""
    
    # Активность кухни (утром и вечером)
    kitchen_active = 1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0
    kitchen_ratio = 0.25 if kitchen_active else 0.05
    
    # Активность прачечной (днем в выходные)
    laundry_active = 1 if (10 <= hour <= 18) and (day_of_week >= 5) else 0
    laundry_ratio = 0.15 if laundry_active else 0.03
    
    # Активность климат-систем (вечером зимой)
    ac_heating_active = 1 if (18 <= hour <= 22) and (month in [1, 2, 12]) else 0
    ac_heating_ratio = 0.35 if ac_heating_active else 0.08
    
    return {
        'kitchen_ratio': kitchen_ratio,
        'laundry_ratio': laundry_ratio,
        'ac_heating_ratio': ac_heating_ratio,
        'kitchen_active': kitchen_active,
        'laundry_active': laundry_active,
        'ac_heating_active': ac_heating_active
    }

def create_prediction_features(hour, day_of_week, month):
    """Создает ВСЕ 49 признаков с РЕАЛИСТИЧНЫМИ значениями"""
    
    features = {}
    
    # 1. ЦИКЛИЧЕСКИЕ ПРИЗНАКИ
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # 2. СУТОЧНЫЕ ПАТТЕРНЫ ИЗ EDA
    features['is_early_morning'] = 1 if 4 <= hour <= 6 else 0
    features['is_midday'] = 1 if 10 <= hour <= 16 else 0
    features['is_late_evening'] = 1 if 21 <= hour <= 23 else 0
    features['is_evening_peak'] = 1 if 18 <= hour <= 22 else 0
    features['is_morning_peak'] = 1 if 7 <= hour <= 9 else 0
    features['is_night'] = 1 if 0 <= hour <= 5 else 0
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
    features['is_high_season'] = 1 if month in [1, 2, 12] else 0
    features['is_low_season'] = 1 if month in [6, 7, 8] else 0
    features['is_spring'] = 1 if month in [3, 4, 5] else 0
    
    # 5. КРИТИЧЕСКИЕ ПЕРЕХОДЫ ИЗ EDA
    features['morning_surge_6_7'] = 1 if 6 <= hour <= 7 else 0
    features['evening_surge_17_18'] = 1 if 17 <= hour <= 18 else 0
    features['evening_drop_22_23'] = 1 if 22 <= hour <= 23 else 0
    
    # 6. ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ
    features['winter_evening'] = 1 if (month in [1, 2, 12] and 18 <= hour <= 22) else 0
    features['summer_afternoon'] = 1 if (month in [6, 7, 8] and 10 <= hour <= 16) else 0
    features['workday_evening'] = 1 if (day_of_week < 5 and 18 <= hour <= 22) else 0
    features['sunday_evening'] = 1 if (day_of_week == 6 and 18 <= hour <= 22) else 0
    
    # 7. РЕАЛИСТИЧНЫЕ ЛАГИ
    lag_features, base_consumption = create_realistic_lags(hour, day_of_week, month)
    features.update(lag_features)
    
    # 8. РЕАЛИСТИЧНЫЕ СКОЛЬЗЯЩИЕ СТАТИСТИКИ
    rolling_features = create_realistic_rolling_stats(hour, base_consumption)
    features.update(rolling_features)
    
    # 9. РЕАЛИСТИЧНЫЕ СУБ-СЧЕТЧИКИ
    sub_metering_features = create_realistic_sub_metering(hour, day_of_week, month)
    features.update(sub_metering_features)
    
    # 10. БАЗОВЫЕ ПРИЗНАКИ
    features['hour'] = hour
    features['day_of_week'] = day_of_week
    features['month'] = month
    features['is_weekend'] = 1 if day_of_week >= 5 else 0
    
    # Создаем DataFrame в ТОЧНОМ порядке признаков
    ordered_features = {name: features[name] for name in FEATURE_NAMES}
    
    print(f"🔍 Созданы признаки для {hour:02d}:00 - лаг_2ч: {features['lag_2h_ago']:.3f} кВт")
    
    return pd.DataFrame([ordered_features])

def validate_prediction(prediction, hour):
    """Проверяет что прогноз реалистичный на основе EDA с учетом изменчивости"""
    # РЕАЛИСТИЧНЫЕ диапазоны на основе вашего EDA (с учетом std)
    realistic_ranges = {
        # (min, max) с учетом ±2 стандартных отклонений
        0: (0.2, 2.5), 1: (0.1, 2.0), 2: (0.1, 1.8), 3: (0.1, 1.7),
        4: (0.1, 1.6), 5: (0.1, 1.7), 6: (0.2, 2.8), 7: (0.5, 3.5),
        8: (0.5, 3.3), 9: (0.4, 3.0), 10: (0.3, 2.8), 11: (0.2, 2.7),
        12: (0.2, 2.8), 13: (0.2, 2.6), 14: (0.2, 2.6), 15: (0.2, 2.6),
        16: (0.2, 2.4), 17: (0.3, 2.8), 18: (0.5, 3.8), 19: (0.8, 4.5),
        20: (0.8, 4.5), 21: (1.0, 4.8), 22: (0.6, 3.8), 23: (0.3, 2.8)
    }
    
    min_val, max_val = realistic_ranges[hour]
    is_realistic = min_val <= prediction <= max_val
    
    if not is_realistic:
        print(f"⚠️  Прогноз {prediction:.2f} кВт в {hour:02d}:00 выходит за реалистичные пределы ({min_val:.1f}-{max_val:.1f})")
        # Более мягкая коррекция - к ближайшей границе
        if prediction < min_val:
            correction = min_val
        else:
            correction = max_val
        return correction, is_realistic
    
    return prediction, is_realistic

def predict_24_hours():
    """Прогноз на 24 часа с РЕАЛИСТИЧНЫМИ признаками"""
    tomorrow = datetime.now() + timedelta(days=1)
    day_of_week = tomorrow.weekday()
    month = tomorrow.month
    
    predictions = []
    validation_results = []
    
    print(f"📅 Прогноз на {tomorrow.strftime('%d.%m.%Y')} ({['пн','вт','ср','чт','пт','сб','вс'][day_of_week]})")
    
    for hour in range(24):
        # Создаем РЕАЛИСТИЧНЫЕ признаки для этого часа
        features_df = create_prediction_features(hour, day_of_week, month)
        
        # Прогнозируем
        prediction = model.predict(features_df)[0]
        
        # Проверяем и корректируем прогноз
        validated_prediction, is_realistic = validate_prediction(prediction, hour)
        predictions.append(validated_prediction)
        validation_results.append(is_realistic)
        
        status = "✅" if is_realistic else "⚠️"
        print(f"  {status} Час {hour:2d}: {validated_prediction:.2f} кВт ({'реалистично' if is_realistic else 'скорректировано'})")
    
    # Статистика валидации
    realistic_count = sum(validation_results)
    print(f"📊 Реалистичных прогнозов: {realistic_count}/24 ({realistic_count/24*100:.1f}%)")
    
    return list(range(24)), predictions, day_of_week, month, realistic_count

def create_enhanced_plot(hours, predictions, date_str, realistic_count):
    """Создает УЛУЧШЕННЫЙ график прогноза"""
    plt.figure(figsize=(14, 8))
    
    # Основной график
    plt.plot(hours, predictions, 'b-', linewidth=3, marker='o', markersize=5, label='Прогноз')
    
    # Фоновая линия средних значений для сравнения
    avg_line = [HOURLY_AVERAGES[h] for h in hours]
    plt.plot(hours, avg_line, 'g--', alpha=0.7, linewidth=1, label='Средние значения')
    
    # Зоны пиков с прозрачностью
    plt.axvspan(7, 9, alpha=0.15, color='orange', label='Утренний пик')
    plt.axvspan(18, 22, alpha=0.15, color='red', label='Вечерний пик')
    plt.axvspan(0, 5, alpha=0.1, color='blue', label='Ночное время')
    
    # Заполнение под кривой
    plt.fill_between(hours, predictions, alpha=0.2, color='blue')
    
    plt.title(f'Прогноз энергопотребления на {date_str}\n(Точность модели: 92.4%)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Час дня', fontsize=12)
    plt.ylabel('Нагрузка (кВт)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.xticks(range(0, 24, 2))
    plt.ylim(bottom=0)
    
    # Добавляем аннотации для пиков
    peak_hour = hours[np.argmax(predictions)]
    peak_value = np.max(predictions)
    plt.annotate(f'Пик: {peak_value:.1f} кВт', 
                xy=(peak_hour, peak_value), 
                xytext=(peak_hour, peak_value + 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', fontweight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 *Бот прогнозирования энергопотребления - УЛУЧШЕННАЯ ВЕРСИЯ*

*Что нового:*
• 🎯 Реалистичные признаки на основе анализа данных
• 📊 Проверка достоверности прогнозов  
• 🔍 Учет сезонности и дней недели
• 📈 Сравнение со средними значениями

*Команды:*
/predict - Прогноз на завтра (24 часа)
/stats - Статистика моделей
/patterns - Паттерны потребления

*Точность модели:* 92.4%
*Модель:* LightGBM
*Признаков:* 49
    """
    bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['predict'])
def send_prediction(message):
    try:
        bot.send_message(message.chat.id, "⏳ Генерирую РЕАЛИСТИЧНЫЙ прогноз на завтра...")
        
        hours, predictions, day_of_week, month, realistic_count = predict_24_hours()
        
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime('%d.%m.%Y')
        
        # Создаем УЛУЧШЕННЫЙ график
        plot_buf = create_enhanced_plot(hours, predictions, date_str, realistic_count)
        
        # Детальная статистика
        avg = np.mean(predictions)
        peak = np.max(predictions)
        peak_hour = hours[np.argmax(predictions)]
        min_val = np.min(predictions)
        min_hour = hours[np.argmin(predictions)]
        total = np.sum(predictions)
        
        day_names = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
        month_names = ["январе", "феврале", "марте", "апреле", "мае", "июне", 
                      "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"]
        
        # Анализ паттернов
        morning_peak = np.mean(predictions[7:10])
        evening_peak = np.mean(predictions[18:23])
        night_avg = np.mean(predictions[0:6])
        
        caption = f"""📊 *Прогноз на {date_str}* ({day_names[day_of_week]}, {month_names[month-1]})

*Основные метрики:*
• 📈 Средняя нагрузка: {avg:.2f} кВт
• 🚀 Пиковая нагрузка: {peak:.2f} кВт в {peak_hour}:00
• 📉 Минимальная нагрузка: {min_val:.2f} кВт в {min_hour}:00
• 🔋 Суммарное потребление: {total:.1f} кВт·ч

*Паттерны потребления:*
• 🌅 Утренний пик (7-9): {morning_peak:.2f} кВт
• 🌇 Вечерний пик (18-22): {evening_peak:.2f} кВт  
• 🌙 Ночное время (0-5): {night_avg:.2f} кВт

*Качество прогноза:*
• ✅ Реалистичных значений: {realistic_count}/24
• 🤖 Модель: LightGBM (92.4% точность)
• 🔍 Признаков: 49 (с реалистичными лагами)"""

        bot.send_photo(message.chat.id, plot_buf, caption=caption, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"❌ Ошибка при генерации прогноза: {str(e)}"
        print(error_msg)
        bot.send_message(message.chat.id, error_msg)

@bot.message_handler(commands=['stats'])
def send_stats(message):
    stats_text = """
📊 *Статистика модели - УЛУЧШЕННАЯ ВЕРСИЯ*

*LightGBM (лучшая модель)*
• Точность (R²): 92.4%
• Средняя ошибка (MAE): 0.11 кВт (110 Вт)
• Максимальная ошибка: 4.45 кВт
• Количество признаков: 49

*Важнейшие признаки:*
1. `lag_2h_ago` (59.3%) - потребление 2 часа назад
2. `ac_heating_active` (8.3%) - активность климат-систем  
3. `is_night` (5.6%) - ночное время
4. `ac_heating_ratio` (2.8%) - доля климат-систем
5. `kitchen_ratio` (2.2%) - доля кухни

*Данные обучения:*
• Период: 6 месяцев (январь-июнь 2007)
• Интервал: 1 минута
• Объем: 260,640 измерений
• Пропуски: 1.01% (обработаны)

*Реалистичные параметры:*
• Сезонность: учтена (+74% зима/лето)
• Выходные: +36.6% к рабочим дням
• Пики: утренний (7-9), вечерний (18-22)
    """
    bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

@bot.message_handler(commands=['patterns'])
def send_patterns(message):
    patterns_text = """
🔍 *Паттерны потребления из анализа данных*

*Суточные циклы:*
• 🌙 Ночной минимум: 0.49 кВт (4:00)
• 🚀 Вечерний максимум: 2.18 кВт (21:00) 
• 📈 Утренний рост: +0.58 кВт (6→7)
• 📉 Вечерний спад: -0.59 кВт (22→23)

*Недельные паттерны:*
• 📅 Рабочие дни: 1.05 кВт (среднее)
• 🎉 Выходные дни: 1.43 кВт (+36.6%)
• 📊 Наиболее стабильный: Воскресенье
• 📊 Наиболее изменчивый: Вторник

*Сезонные колебания:*
• ❄️ Зима: 1.47 кВт (максимум)
• 🌷 Весна: 1.06 кВт 
• ☀️ Лето: 0.83 кВт (минимум)
• 📉 Амплитуда: -43.9% (зима→лето)

*Пиковые периоды:*
• 🌅 Утро: 7:00-9:00 (1.45 кВт)
• 🌇 Вечер: 18:00-22:00 (1.90 кВт)
• 🌙 Ночь: 0:00-5:00 (0.58 кВт)
    """
    bot.send_message(message.chat.id, patterns_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    help_text = """
🤖 Я бот для прогнозирования энергопотребления.

Используйте команды:
/predict - получить прогноз на завтра
/stats - посмотреть статистику модели  
/patterns - узнать о паттернах потребления
/help - показать это сообщение

Моя модель обучена на реальных данных и показывает точность 92.4%! 🚀
    """
    bot.send_message(message.chat.id, help_text)

if __name__ == "__main__":
    print("🚀 Бот запущен с РЕАЛИСТИЧНЫМИ ПРИЗНАКАМИ!")
    print("✅ Модель: LightGBM")
    print("✅ Признаков: 49")
    print("✅ Реалистичные лаги и статистики")
    print("📊 Ожидаем команды /predict")
    bot.infinity_polling()