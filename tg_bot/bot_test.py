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

from data_loader import RealisticDataGenerator
from feature_generator import HonestFeatureGenerator

from feature_generator import SimpleFeatureGenerator
feature_gen = SimpleFeatureGenerator()

# Загружаем токен
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Загружаем модель и признаки
try:
    model = joblib.load('models/lightgbm_best_model.pkl')
    with open('models/feature_names.json', 'r', encoding='utf-8') as f:
        FEATURE_NAMES = json.load(f)
    print(f"✅ Модель загружена. Ожидает {len(FEATURE_NAMES)} признаков")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
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

class RealDataGenerator:
    def __init__(self):
        self.data_loader = RealisticDataGenerator()
        self.historical_predictions = {}
    
    def get_historical_consumption(self, hour, day_of_week, month, target_date):
        """Использует РЕАЛЬНЫЕ исторические данные вместо генерации"""
        return self.data_loader.get_historical_pattern(target_date, hour)
    
    
    def get_realistic_rolling_stats(self, hour, month, target_date):
        """Скользящие статистики на основе реальных данных"""
        # Используем исторические данные для расчёта статистик
        try:
            # Среднее за последние 3 часа из исторических данных
            recent_hours = []
            for i in range(3):
                hist_date = target_date - timedelta(hours=i+1)
                hist_value = self.data_loader.get_historical_pattern(hist_date, hour)
                recent_hours.append(hist_value)
            
            rolling_3h = np.mean(recent_hours) if recent_hours else REAL_HOURLY_AVERAGES[hour]
        except:
            rolling_3h = REAL_HOURLY_AVERAGES[hour]
        
        # 24-часовое среднее из EDA
        all_hourly_values = list(REAL_HOURLY_AVERAGES.values())
        rolling_24h = np.mean(all_hourly_values)
        
        return {
            'rolling_mean_3h_past': max(0.1, rolling_3h),
            'rolling_mean_24h': max(0.1, rolling_24h),
            'rolling_mean_7d_past': max(0.1, rolling_24h * 0.995),
            'rolling_mean_168h': max(0.1, rolling_24h * 0.99)
        }
    
    def get_realistic_submetering(self, hour, day_of_week, month):
        """Оставляем детерминированную версию (она нормальная)"""
        # Твой текущий код остаётся без изменений
        kitchen_active = 1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0
        laundry_active = 1 if (10 <= hour <= 18) and (day_of_week >= 5) else 0
        ac_heating_active = 1 if ((18 <= hour <= 22) and (month in [12, 1, 2])) or \
                                ((13 <= hour <= 17) and (month in [6, 7, 8])) else 0
        
        hour_factor = hour / 24.0
        day_factor = day_of_week / 7.0
        
        if kitchen_active:
            kitchen_ratio = 0.15 + (hour_factor * 0.1)
        else:
            kitchen_ratio = 0.02 + (hour_factor * 0.06)
            
        if laundry_active:
            laundry_ratio = 0.08 + (day_factor * 0.07)
        else:
            laundry_ratio = 0.01 + (hour_factor * 0.03)
            
        if ac_heating_active:
            ac_heating_ratio = 0.25 + (hour_factor * 0.1)
        else:
            ac_heating_ratio = 0.05 + (day_factor * 0.07)
        
        return {
            'kitchen_ratio': kitchen_ratio,
            'laundry_ratio': laundry_ratio,
            'ac_heating_ratio': ac_heating_ratio,
            'kitchen_active': kitchen_active,
            'laundry_active': laundry_active,
            'ac_heating_active': ac_heating_active
        }


# ⚡ ЗАМЕНИ генератор на новый
data_gen = RealisticDataGenerator()
feature_gen = HonestFeatureGenerator(data_gen)


# Временно используем старую функцию но убираем из нее проблемные части
def create_realistic_features_NO_LEAKAGE(hour, day_of_week, month, target_date):
    """Временное решение - используем старую функцию но без утечек"""
    
    # ⚡ ВМЕСТО ГЕНЕРАЦИИ - используем исторические данные
    current_consumption = data_gen.get_historical_consumption(hour, day_of_week, month, target_date)
    
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

    # 7. РЕАЛЬНЫЕ ЛАГИ БЕЗ УТЕЧЕК
    lag_features = data_gen.data_loader.get_realistic_lags_NO_LEAKAGE(hour, target_date)
    features.update(lag_features)
    
    # 8. РЕАЛЬНЫЕ СКОЛЬЗЯЩИЕ СТАТИСТИКИ БЕЗ УТЕЧЕК
    rolling_features = data_gen.data_loader.get_realistic_rolling_stats(hour, month, target_date)
    features.update(rolling_features)
    
    # 9. СУБ-СЧЕТЧИКИ
    submetering_features = data_gen.get_realistic_submetering(hour, day_of_week, month)
    features.update(submetering_features)
    
    # 10. БАЗОВЫЕ ПРИЗНАКИ
    features['hour'] = hour
    features['day_of_week'] = day_of_week
    features['month'] = month
    features['is_weekend'] = 1 if day_of_week >= 5 else 0
    
    # Создаем DataFrame в правильном порядке
    ordered_features = {name: features[name] for name in FEATURE_NAMES}
    
    return pd.DataFrame([ordered_features])


def correct_prediction_scale(predictions, day_of_week, month):
    """Корректируем масштаб прогнозов к реалистичным значениям"""
    # Вычисляем текущий масштаб прогнозов
    current_mean = np.mean(predictions)
    
    # Целевой масштаб из реальных данных EDA
    target_mean = np.mean(list(REAL_HOURLY_AVERAGES.values()))
    
    # Сезонные поправки
    if month in [12, 1, 2]: target_mean *= 1.2      # зима
    elif month in [6, 7, 8]: target_mean *= 0.8     # лето
    
    # Поправка на выходные
    if day_of_week >= 5: target_mean *= 1.3
    
    # Коэффициент коррекции
    scale_factor = target_mean / current_mean if current_mean > 0 else 0.3
    
    # Применяем коррекцию
    corrected = [p * scale_factor for p in predictions]
    
    print(f"🔧 Коррекция масштаба: {current_mean:.2f} → {target_mean:.2f} (×{scale_factor:.2f})")
    
    return corrected


# Упрощенная функция прогноза:
def predict_honest(target_date):
    """ЧЕСТНЫЙ прогноз без утечек данных"""
    day_of_week = target_date.weekday()
    month = target_date.month
    
    predictions = []
    
    print(f"🎯 ЧЕСТНЫЙ прогноз на {target_date.strftime('%d.%m.%Y')}")
    print("=" * 50)
    
    for hour in range(24):
        # ⚡ ИСПОЛЬЗУЕМ НОВЫЙ ГЕНЕРАТОР БЕЗ УТЕЧЕК
        features_df = feature_gen.create_honest_features(hour, day_of_week, month, target_date)
        
        prediction = model.predict(features_df)[0]
        predictions.append(max(0.1, prediction))
        
        print(f"  {hour:2d}:00 -> {prediction:.2f} кВт")
    
    avg_pred = np.mean(predictions)
    print(f"📊 Средний прогноз: {avg_pred:.2f} кВт")
    print("=" * 50)
    
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
    plt.ylim(bottom=0, top=3)
    
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
        hours, predictions, day_of_week, month = predict_honest(target_date)
        
        date_str = target_date.strftime('%d.%m.%Y')
        day_names = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        
        # Создаем график
        plt.figure(figsize=(12, 6))
        plt.plot(hours, predictions, 'b-', linewidth=2, marker='o', label='Прогноз ML')
        plt.plot(hours, [REAL_HOURLY_AVERAGES[h] for h in hours], 'r--', label='Реальные средние')
        plt.title(f'Прогноз на {date_str} ({day_names[day_of_week]})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
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
        hours, pred_tomorrow, dow_tomorrow, month_tomorrow = predict_honest(tomorrow)
        
        # Прогноз на послезавтра
        day_after = datetime.now() + timedelta(days=2)
        _, pred_day_after, dow_day_after, month_day_after = predict_honest(day_after)
        
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
• LightGBM R²: 92.4% (на тестовых данных)
• Средняя ошибка: 0.11 кВт
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

if __name__ == "__main__":
    print("🚀 Бот запущен для ЧЕСТНОЙ оценки модели!")
    print("📊 Кнопки для сравнения прогнозов активированы")
    print("⚠️  Ожидаем выявления реальных проблем модели")
    bot.infinity_polling()