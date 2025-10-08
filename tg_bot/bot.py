# import telebot
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import os
# import sys
# import io
# import json

# # Настройка путей
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from tg_bot.config import BOT_TOKEN, MODEL_PATHS, check_config
# from tg_bot.feature_engineer import FeatureEngineer
# from tg_bot.eda_patterns import HOURLY_PATTERNS

# # Инициализация
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

# print("🔄 Проверка конфигурации...")
# available_models = check_config()

# # Загрузка моделей и инициализация FeatureEngineer
# models = {}
# feature_engineer = FeatureEngineer()

# for model_name, model_path in MODEL_PATHS.items():
#     try:
#         if model_name in available_models:
#             models[model_name] = joblib.load(model_path)
#             print(f"✅ {model_name} успешно загружена!")
#     except Exception as e:
#         print(f"❌ Ошибка загрузки {model_name}: {e}")

# bot = telebot.TeleBot(BOT_TOKEN)

# # Загрузка всех доступных моделей
# models = {}
# for model_name, model_path in MODEL_PATHS.items():
#     try:
#         if model_name in available_models:
#             models[model_name] = joblib.load(model_path)
#             print(f"✅ {model_name} успешно загружена!")
#         else:
#             print(f"❌ {model_name} пропущена - файл не найден")
#     except Exception as e:
#         print(f"❌ Ошибка загрузки {model_name}: {e}")

# print(f"📊 Загружено моделей: {len(models)}")

# def load_feature_names():
#     """Загружает имена признаков из файла"""
#     try:
#         with open('models/feature_names.json', 'r', encoding='utf-8') as f:
#             feature_names = json.load(f)
#         print(f"✅ Загружено {len(feature_names)} признаков из файла")
#         return feature_names
#     except Exception as e:
#         print(f"❌ Ошибка загрузки признаков: {e}")
#         return None

# def create_features_from_template(hour, day_of_week, month):
#     """Создает признаки с РЕАЛЬНЫМИ паттернами из EDA"""
#     feature_names = load_feature_names()
    
#     # РЕАЛЬНЫЕ ЗНАЧЕНИЯ ИЗ EDA
#     current_hour_consumption = get_hourly_pattern(hour, 'mean')
#     current_day_consumption = get_daily_pattern(day_of_week, 'mean')
#     current_month_consumption = get_monthly_pattern(month, 'mean')
    
#     base_values = {
#         'Global_reactive_power': 0.1,
#         'Voltage': 240.0,
#         'Global_intensity': 2.5,
#         'Sub_metering_1': 0.0,
#         'Sub_metering_2': 0.0,
#         'Sub_metering_3': 0.0,
#         'hour': hour,
#         'day_of_week': day_of_week,
#         'month': month,
#         'is_weekend': 1 if day_of_week >= 5 else 0,
        
#         # ✅ РЕАЛЬНЫЕ ЛАГИ ИЗ EDA
#         'lag_2h_ago': get_hourly_pattern((hour - 2) % 24, 'mean'),
#         'lag_6h_ago': get_hourly_pattern((hour - 6) % 24, 'mean'), 
#         'lag_12h_ago': get_hourly_pattern((hour - 12) % 24, 'mean'),
#         'lag_same_day_24h': current_hour_consumption,  # Вчера в это же время
#         'lag_week_ago_168h': current_hour_consumption, # Неделю назад
        
#         # ✅ РЕАЛЬНЫЕ СКОЛЬЗЯЩИЕ СТАТИСТИКИ
#         'rolling_mean_3h_past': np.mean([
#             get_hourly_pattern((hour - 1) % 24, 'mean'),
#             get_hourly_pattern((hour - 2) % 24, 'mean'), 
#             get_hourly_pattern((hour - 3) % 24, 'mean')
#         ]),
#         'rolling_mean_24h_past': np.mean(list(HOURLY_PATTERNS['mean'].values())),
#         'rolling_std_24h_past': np.std(list(HOURLY_PATTERNS['mean'].values())),
#         'rolling_mean_7d_past': np.mean(list(WEEKLY_PATTERNS['mean'].values())),
        
#         # Циклические признаки
#         'hour_sin': np.sin(2 * np.pi * hour/24),
#         'hour_cos': np.cos(2 * np.pi * hour/24),
#         'month_sin': np.sin(2 * np.pi * month/12),
#         'month_cos': np.cos(2 * np.pi * month/12),
#         'day_of_week_sin': np.sin(2 * np.pi * day_of_week/7),
#         'day_of_week_cos': np.cos(2 * np.pi * day_of_week/7),
        
#         # ✅ РЕАЛЬНЫЕ БИНАРНЫЕ ПРИЗНАКИ
#         'is_early_morning': 1 if 4 <= hour <= 6 else 0,
#         'is_midday': 1 if 10 <= hour <= 16 else 0,
#         'is_evening_peak': 1 if 18 <= hour <= 22 else 0,
#         'is_morning_peak': 1 if 7 <= hour <= 9 else 0,
#         'is_night': 1 if is_night_hour(hour) else 0,
#         'is_deep_night': 1 if 1 <= hour <= 4 else 0,
        
#         # ✅ ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ НА ОСНОВЕ EDA
#         'is_week_start': 1 if day_of_week in [0, 1] else 0,  # Пн-Вт
#         'is_week_end': 1 if day_of_week in [4, 5] else 0,    # Пт-Сб
#         'is_high_season': 1 if month in [1, 2, 12] else 0,   # Зима
#         'is_low_season': 1 if month in [6, 7, 8] else 0,     # Лето
#     }
    
#     # Остальной код без изменений...
#     features = {}
#     for feature_name in feature_names:
#         if feature_name in base_values:
#             features[feature_name] = base_values[feature_name]
#         elif 'is_' in feature_name or 'peak' in feature_name or 'night' in feature_name:
#             features[feature_name] = 0
#         elif 'ratio' in feature_name:
#             features[feature_name] = 0.0
#         else:
#             features[feature_name] = 0.0
    
#     # Вычисляем отношения для суб-счетчиков
#     total_sub = features.get('Sub_metering_1', 0) + features.get('Sub_metering_2', 0) + features.get('Sub_metering_3', 0)
#     if 'kitchen_ratio' in features:
#         features['kitchen_ratio'] = features.get('Sub_metering_1', 0) / (total_sub + 0.001)
#     if 'laundry_ratio' in features:
#         features['laundry_ratio'] = features.get('Sub_metering_2', 0) / (total_sub + 0.001)
#     if 'ac_heating_ratio' in features:
#         features['ac_heating_ratio'] = features.get('Sub_metering_3', 0) / (total_sub + 0.001)
    
#     print(f"✅ Создано {len(features)} признаков с РЕАЛЬНЫМИ EDA паттернами")
#     return pd.DataFrame([features])

# def scale_predictions_to_reality(prediction, hour):
#     """Масштабирует прогноз к реалистичным значениям на основе EDA"""
#     target_pattern = {
#         0: 0.78, 1: 0.63, 2: 0.54, 3: 0.52, 4: 0.49, 5: 0.53,
#         6: 0.94, 7: 1.52, 8: 1.49, 9: 1.34, 10: 1.20, 11: 1.10,
#         12: 1.05, 13: 1.00, 14: 1.04, 15: 1.00, 16: 0.95, 17: 1.07,
#         18: 1.50, 19: 2.07, 20: 2.07, 21: 2.18, 22: 1.67, 23: 1.08
#     }
    
#     target = target_pattern.get(hour, 1.0)
    
#     if prediction < 0.5:
#         scaled_prediction = target * 0.8
#     else:
#         scale_factor = target / 1.0
#         scaled_prediction = prediction * scale_factor
    
#     return max(0.1, min(3.0, scaled_prediction))

# def generate_tomorrow_predictions():
#     """Генерирует прогноз на завтра (24 часа)"""
#     tomorrow = datetime.now() + timedelta(days=1)
#     day_of_week = tomorrow.weekday()
#     month = tomorrow.month
    
#     model_name = 'lightgbm' if 'lightgbm' in models else list(models.keys())[0]
#     model = models[model_name]
    
#     predictions = []
#     hours = []
    
#     print(f"🔍 Прогноз на {tomorrow.strftime('%d.%m.%Y')} ({['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][day_of_week]})")
    
#     for hour in range(24):
#         features_df = create_features_from_template(hour, day_of_week, month)
        
#         try:
#             prediction = model.predict(features_df)[0]
#             scaled_prediction = scale_predictions_to_reality(prediction, hour)
            
#             predictions.append(scaled_prediction)
#             hours.append(hour)
#             print(f"   Час {hour:2d}: {scaled_prediction:.3f} кВт (исходный: {prediction:.3f} кВт)")
#         except Exception as e:
#             print(f"❌ Ошибка при прогнозе для часа {hour}: {e}")
#             if 0 <= hour <= 5: base_consumption = 0.5
#             elif 7 <= hour <= 9: base_consumption = 1.5  
#             elif 18 <= hour <= 22: base_consumption = 2.0
#             else: base_consumption = 1.0
#             predictions.append(base_consumption)
#             hours.append(hour)

#     print(f"✅ Прогноз завершен. Среднее: {np.mean(predictions):.3f} кВт")
#     return hours, predictions, model_name, day_of_week, month

# def create_tomorrow_plot(hours, predictions, model_name, date_str):
#     """Создает график прогноза на завтра"""
#     plt.style.use('seaborn-v0_8')
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Основной график
#     ax.plot(hours, predictions, linewidth=3, alpha=0.8, color='#2E86AB', 
#             marker='o', markersize=4, label='Прогноз нагрузки')
    
#     # Зоны пиков
#     ax.axvspan(7, 9, alpha=0.2, color='#F9C74F', label='Утренний пик 7:00-9:00')
#     ax.axvspan(18, 22, alpha=0.2, color='#F94144', label='Вечерний пик 18:00-22:00')
#     ax.axvspan(0, 5, alpha=0.2, color='#577590', label='Ночное время 0:00-5:00')
    
#     # Настройки графика
#     ax.set_xlabel('Час дня', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Нагрузка (кВт)', fontsize=12, fontweight='bold')
#     ax.set_title(f'Прогноз энергопотребления на {date_str}', 
#                 fontsize=14, fontweight='bold', pad=20)
    
#     # Сетка и оформление
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.legend(loc='upper right', framealpha=0.9)
#     ax.set_xticks(range(0, 24, 2))
#     ax.set_xlim(0, 23)
    
#     # Выделяем пик
#     max_idx = np.argmax(predictions)
#     ax.plot(max_idx, predictions[max_idx], 'ro', markersize=8)
#     ax.text(max_idx, predictions[max_idx] + 0.2, 
#            f'Пик: {predictions[max_idx]:.2f} кВт', 
#            ha='center', fontweight='bold', fontsize=10)
    
#     plt.tight_layout()
    
#     # Сохраняем в буфер
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
#                 facecolor='white', edgecolor='none')
#     buf.seek(0)
#     plt.close(fig)
    
#     return buf

# def calculate_metrics(predictions):
#     """Рассчитывает метрики для прогнозов"""
#     return {
#         'min': np.min(predictions),
#         'max': np.max(predictions),
#         'mean': np.mean(predictions),
#         'total': np.sum(predictions),
#         'morning_peak': np.mean(predictions[7:10]),
#         'evening_peak': np.mean(predictions[18:23]),
#         'night': np.mean(predictions[0:6])
#     }

# @bot.message_handler(commands=['start'])
# def send_welcome(message):
#     """Главное меню с кнопкой прогноза"""
#     markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
#     markup.add('📊 Прогноз на завтра')
#     markup.add('📈 Статистика моделей', 'ℹ️ Помощь')
    
#     welcome_text = """
# 🤖 *Бот прогнозирования энергопотребления*

# Просто нажмите кнопку *"Прогноз на завтра"* чтобы получить прогноз нагрузки на следующие 24 часа.

# *Точность прогноза:* 92.4%
# *Используемая модель:* LightGBM
#     """
    
#     bot.send_message(message.chat.id, welcome_text, 
#                    reply_markup=markup, parse_mode='Markdown')

# @bot.message_handler(commands=['help'])
# def send_help(message):
#     """Помощь"""
#     help_text = """
# ℹ️ *Помощь по боту*

# Этот бот прогнозирует энергопотребление на основе машинного обучения.

# *Как использовать:*
# 1. Нажмите кнопку "Прогноз на завтра"
# 2. Получите график и метрики нагрузки

# *Модели:*
# • LightGBM - 92.4% точности
# • XGBoost - 92.2% точности  
# • RandomForest - 90.6% точности

# *Данные обучения:*
# 6 месяцев реальных данных энергопотребления
#     """
#     bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

# @bot.message_handler(commands=['stats'])
# def send_stats(message):
#     """Статистика моделей"""
#     stats_text = """
# 📊 *Статистика моделей*

# *LightGBM* 🚀 (основная)
# • Точность: 92.4%
# • Ошибка: 0.11 кВт
# • Обучена на 260k+ записях

# *XGBoost* ⚡
# • Точность: 92.2%
# • Ошибка: 0.11 кВт

# *RandomForest* 🌲
# • Точность: 90.6%
# • Ошибка: 0.12 кВт

# Все модели обучены на 6-месячных данных с минутным интервалом.
#     """
#     bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

# @bot.message_handler(func=lambda message: True)
# def handle_all_messages(message):
#     """Обработка всех сообщений и кнопок"""
#     if message.text == '📊 Прогноз на завтра':
#         generate_tomorrow_prediction(message)
#     elif message.text == '📈 Статистика моделей':
#         send_stats(message)
#     elif message.text == 'ℹ️ Помощь':
#         send_help(message)
#     else:
#         send_welcome(message)

# def generate_tomorrow_prediction(message):
#     """Генерирует и отправляет прогноз на завтра"""
#     try:
#         bot.send_message(message.chat.id, "⏳ *Генерирую прогноз на завтра...*", parse_mode='Markdown')
        
#         # Генерируем прогноз
#         hours, predictions, model_name, day_of_week, month = generate_tomorrow_predictions()
#         metrics = calculate_metrics(predictions)
        
#         # Форматируем дату для заголовка
#         tomorrow = datetime.now() + timedelta(days=1)
#         date_str = tomorrow.strftime('%d.%m.%Y')
        
#         # Создаем график С ДАТОЙ
#         plot_buf = create_tomorrow_plot(hours, predictions, model_name, date_str)
        
#         # Названия дней и месяцев
#         day_names = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
#         month_names = ["", "январе", "феврале", "марте", "апреле", "мае", "июне", 
#                       "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"]
        
#         # Формируем текст с метриками (ИСПРАВЛЕННЫЙ - убрал лишние звездочки)
#         metrics_text = f"""📊 *ПРОГНОЗ НА ЗАВТРА* ({day_names[day_of_week].capitalize()}, {month_names[month]})

# *Основные метрики:*
# • 🎯 Средняя нагрузка: {metrics['mean']:.2f} кВт
# • 📈 Пиковая нагрузка: {metrics['max']:.2f} кВт
# • 📉 Минимальная нагрузка: {metrics['min']:.2f} кВт
# • 🔋 Суммарное потребление: {metrics['total']:.2f} кВт·ч

# *Пиковые периоды:*
# • 🌅 Утренний пик (7-9): {metrics['morning_peak']:.2f} кВт
# • 🌇 Вечерний пик (18-22): {metrics['evening_peak']:.2f} кВт 
# • 🌙 Ночное время (0-5): {metrics['night']:.2f} кВт

# 💡 *Точность модели: 92.4%*
# 🤖 *Использована модель: {model_name.upper()}*"""
        
#         # Отправляем график и метрики
#         bot.send_photo(message.chat.id, plot_buf, caption=metrics_text, parse_mode='Markdown')
        
#         # Добавляем кнопку для нового прогноза
#         markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
#         markup.add('📊 Новый прогноз', '📈 Статистика')
        
#         bot.send_message(
#             message.chat.id,
#             "🔄 *Хотите сделать еще один прогноз?*",
#             reply_markup=markup,
#             parse_mode='Markdown'
#         )
        
#     except Exception as e:
#         error_text = f"""❌ *Произошла ошибка при генерации прогноза*

# Ошибка: {str(e)}

# Пожалуйста, попробуйте еще раз или обратитесь к администратору."""
#         bot.send_message(message.chat.id, error_text, parse_mode='Markdown')
#         print(f"Error in generate_tomorrow_prediction: {e}")

# def main():
#     """Запуск бота"""
#     print("🤖 Запуск бота прогнозирования энергопотребления...")
#     print("✅ Бот готов к работе!")
#     print("⏹️  Для остановки нажмите Ctrl+C")
    
#     try:
#         bot.infinity_polling()
#     except Exception as e:
#         print(f"❌ Ошибка при работе бота: {e}")

# if __name__ == "__main__":
#     main()



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

def create_prediction_features(hour, day_of_week, month):
    """Создает ВСЕ 49 признаков в ТОЧНОМ порядке как при обучении"""
    
    features = {}
    
    # 1. ЦИКЛИЧЕСКИЕ ПРИЗНАКИ
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # 2. ВРЕМЕННЫЕ ПАТТЕРНЫ
    features['is_early_morning'] = 1 if 4 <= hour <= 6 else 0
    features['is_midday'] = 1 if 10 <= hour <= 16 else 0
    features['is_late_evening'] = 1 if 21 <= hour <= 23 else 0
    features['is_evening_peak'] = 1 if 18 <= hour <= 22 else 0
    features['is_morning_peak'] = 1 if 7 <= hour <= 9 else 0
    features['is_night'] = 1 if 0 <= hour <= 5 else 0
    features['is_deep_night'] = 1 if 1 <= hour <= 4 else 0
    
    # 3. ДНИ НЕДЕЛИ
    features['is_monday'] = 1 if day_of_week == 0 else 0
    features['is_friday'] = 1 if day_of_week == 4 else 0
    features['is_sunday'] = 1 if day_of_week == 6 else 0
    features['is_week_start'] = 1 if day_of_week in [0, 1] else 0
    features['is_week_end'] = 1 if day_of_week in [4, 5] else 0
    features['weekend_evening_boost'] = 1 if (day_of_week >= 5 and 18 <= hour <= 22) else 0
    features['weekend_morning'] = 1 if (day_of_week >= 5 and 7 <= hour <= 9) else 0
    
    # 4. СЕЗОННЫЕ
    features['is_high_season'] = 1 if month in [1, 2, 12] else 0  # Зима
    features['is_low_season'] = 1 if month in [6, 7, 8] else 0    # Лето
    features['is_spring'] = 1 if month in [3, 4, 5] else 0
    
    # 5. КРИТИЧЕСКИЕ ПЕРЕХОДЫ
    features['morning_surge_6_7'] = 1 if 6 <= hour <= 7 else 0
    features['evening_surge_17_18'] = 1 if 17 <= hour <= 18 else 0
    features['evening_drop_22_23'] = 1 if 22 <= hour <= 23 else 0
    
    # 6. ВЗАИМОДЕЙСТВИЯ
    features['winter_evening'] = 1 if (month in [1, 2, 12] and 18 <= hour <= 22) else 0
    features['summer_afternoon'] = 1 if (month in [6, 7, 8] and 10 <= hour <= 16) else 0
    features['workday_evening'] = 1 if (day_of_week < 5 and 18 <= hour <= 22) else 0
    features['sunday_evening'] = 1 if (day_of_week == 6 and 18 <= hour <= 22) else 0
    
    # 7. ЛАГИ (используем средние значения по часам)
    # Простая эвристика: ночь=0.5, утро=1.5, день=1.0, вечер=2.0
    base_consumption = 1.0
    if 0 <= hour <= 5: base_consumption = 0.5
    elif 7 <= hour <= 9: base_consumption = 1.5  
    elif 18 <= hour <= 22: base_consumption = 2.0
    
    features['lag_same_day_24h'] = base_consumption
    features['lag_week_ago_168h'] = base_consumption
    features['lag_2h_ago'] = base_consumption
    features['lag_6h_ago'] = base_consumption
    features['lag_12h_ago'] = base_consumption
    
    # 8. СКОЛЬЗЯЩИЕ СТАТИСТИКИ
    features['rolling_mean_24h_past'] = base_consumption
    features['rolling_std_24h_past'] = 0.5
    features['rolling_mean_7d_past'] = base_consumption
    features['rolling_mean_3h_past'] = base_consumption
    
    # 9. СУБ-СЧЕТЧИКИ
    features['kitchen_ratio'] = 0.2 if (7 <= hour <= 9 or 18 <= hour <= 20) else 0.05
    features['laundry_ratio'] = 0.1 if (10 <= hour <= 18 and day_of_week >= 5) else 0.02
    features['ac_heating_ratio'] = 0.3 if (18 <= hour <= 22 and month in [1, 2, 12]) else 0.1
    
    features['kitchen_active'] = 1 if (7 <= hour <= 9 or 18 <= hour <= 20) else 0
    features['laundry_active'] = 1 if (10 <= hour <= 18 and day_of_week >= 5) else 0
    features['ac_heating_active'] = 1 if (18 <= hour <= 22 and month in [1, 2, 12]) else 0
    
    # 10. БАЗОВЫЕ ПРИЗНАКИ
    features['hour'] = hour
    features['day_of_week'] = day_of_week
    features['month'] = month
    features['is_weekend'] = 1 if day_of_week >= 5 else 0
    
    # Создаем DataFrame в ТОЧНОМ порядке признаков
    ordered_features = {name: features[name] for name in FEATURE_NAMES}
    return pd.DataFrame([ordered_features])

def predict_24_hours():
    """Прогноз на 24 часа"""
    tomorrow = datetime.now() + timedelta(days=1)
    day_of_week = tomorrow.weekday()
    month = tomorrow.month
    
    predictions = []
    
    for hour in range(24):
        # Создаем признаки для этого часа
        features_df = create_prediction_features(hour, day_of_week, month)
        
        # Прогнозируем
        prediction = model.predict(features_df)[0]
        prediction = max(0.1, min(5.0, prediction))  # Ограничиваем разумными значениями
        
        predictions.append(prediction)
        print(f"Час {hour:2d}: {prediction:.2f} кВт")
    
    return list(range(24)), predictions, day_of_week, month

def create_plot(hours, predictions, date_str):
    """Создает график прогноза"""
    plt.figure(figsize=(12, 6))
    
    # Основной график
    plt.plot(hours, predictions, 'b-', linewidth=3, marker='o', markersize=4)
    
    # Зоны пиков
    plt.axvspan(7, 9, alpha=0.2, color='orange', label='Утренний пик')
    plt.axvspan(18, 22, alpha=0.2, color='red', label='Вечерний пик')
    plt.axvspan(0, 5, alpha=0.2, color='blue', label='Ночное время')
    
    plt.title(f'Прогноз энергопотребления на {date_str}', fontsize=14, fontweight='bold')
    plt.xlabel('Час дня')
    plt.ylabel('Нагрузка (кВт)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, 24, 2))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 *Бот прогнозирования энергопотребления*

*Команды:*
/predict - Прогноз на завтра (24 часа)
/stats - Статистика моделей

*Точность модели:* 92.4%
*Модель:* LightGBM
    """
    bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['predict'])
def send_prediction(message):
    try:
        bot.send_message(message.chat.id, "⏳ Генерирую прогноз на завтра...")
        
        hours, predictions, day_of_week, month = predict_24_hours()
        
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime('%d.%m.%Y')
        
        # Создаем график
        plot_buf = create_plot(hours, predictions, date_str)
        
        # Статистика
        avg = np.mean(predictions)
        peak = np.max(predictions)
        peak_hour = hours[np.argmax(predictions)]
        total = np.sum(predictions)
        
        day_names = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
        
        caption = f"""📊 *Прогноз на {date_str}* ({day_names[day_of_week]})

*Основные метрики:*
• 📈 Средняя нагрузка: {avg:.2f} кВт
• 🚀 Пиковая нагрузка: {peak:.2f} кВт в {peak_hour}:00
• 📉 Минимальная нагрузка: {np.min(predictions):.2f} кВт
• 🔋 Суммарное потребление: {total:.1f} кВт·ч

*Модель:* LightGBM (92.4% точность)"""
        
        bot.send_photo(message.chat.id, plot_buf, caption=caption, parse_mode='Markdown')
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка: {str(e)}")

@bot.message_handler(commands=['stats'])
def send_stats(message):
    stats_text = """
📊 *Статистика модели*

*LightGBM*
• Точность (R²): 92.4%
• Средняя ошибка (MAE): 0.11 кВт
• Обучена на: 260,640 записях
• Количество признаков: 49

*Данные обучения:*
• Период: 6 месяцев (январь-июнь 2007)
• Интервал: 1 минута
• Объем: 260k+ измерений
    """
    bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

if __name__ == "__main__":
    print("✅ Бот запущен! Ожидаем команды /predict")
    bot.infinity_polling()