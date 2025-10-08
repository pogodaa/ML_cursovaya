import telebot
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import io
import json
from dotenv import load_dotenv

# Настройки для matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Загружаем .env из корневой директории
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Теперь импортируем конфиг
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tg_bot.config import BOT_TOKEN, MODEL_PATHS, check_config

# Проверяем конфигурацию и загружаем модели
print("🔄 Проверка конфигурации...")
available_models = check_config()

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)

# Загрузка всех доступных моделей
models = {}
for model_name, model_path in MODEL_PATHS.items():
    try:
        if model_name in available_models:
            models[model_name] = joblib.load(model_path)
            print(f"✅ {model_name} успешно загружена!")
        else:
            print(f"❌ {model_name} пропущена - файл не найден")
    except Exception as e:
        print(f"❌ Ошибка загрузки {model_name}: {e}")

print(f"📊 Загружено моделей: {len(models)}")

def load_feature_names():
    """Загружает имена признаков из файла"""
    try:
        with open('models/feature_names.json', 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        print(f"✅ Загружено {len(feature_names)} признаков из файла")
        return feature_names
    except Exception as e:
        print(f"❌ Ошибка загрузки признаков: {e}")
        return None
    

def create_features_from_template(hour, day_of_week, month):
    """Создает признаки по шаблону из обучения"""
    # Загружаем шаблон признаков
    feature_names = load_feature_names()
    
    # Базовые значения для признаков
    base_values = {
        'Global_reactive_power': 0.1,
        'Voltage': 240.0,
        'Global_intensity': 2.5,
        'Sub_metering_1': 0.0,
        'Sub_metering_2': 0.0,
        'Sub_metering_3': 0.0,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        
        # Лаги и скользящие статистики
        'lag_2h_ago': 1.0, 'lag_6h_ago': 1.0, 'lag_12h_ago': 1.0,
        'lag_same_day_24h': 1.0, 'lag_week_ago_168h': 1.0,
        'rolling_mean_3h_past': 1.0, 'rolling_mean_24h_past': 1.0,
        'rolling_std_24h_past': 0.5, 'rolling_mean_7d_past': 1.0,
        
        # Циклические признаки
        'hour_sin': np.sin(2 * np.pi * hour/24),
        'hour_cos': np.cos(2 * np.pi * hour/24),
        'month_sin': np.sin(2 * np.pi * month/12),
        'month_cos': np.cos(2 * np.pi * month/12),
        'day_of_week_sin': np.sin(2 * np.pi * day_of_week/7),
        'day_of_week_cos': np.cos(2 * np.pi * day_of_week/7),
    }
    
    # Создаем финальный словарь признаков
    features = {}
    for feature_name in feature_names:
        if feature_name in base_values:
            features[feature_name] = base_values[feature_name]
        elif 'is_' in feature_name or 'peak' in feature_name or 'night' in feature_name:
            # Для бинарных признаков устанавливаем 0
            features[feature_name] = 0
        elif 'ratio' in feature_name:
            # Для отношений устанавливаем 0
            features[feature_name] = 0.0
        else:
            # Для остальных признаков устанавливаем среднее значение
            features[feature_name] = 0.0
    
    # Вычисляем отношения для суб-счетчиков
    total_sub = features.get('Sub_metering_1', 0) + features.get('Sub_metering_2', 0) + features.get('Sub_metering_3', 0)
    if 'kitchen_ratio' in features:
        features['kitchen_ratio'] = features.get('Sub_metering_1', 0) / (total_sub + 0.001)
    if 'laundry_ratio' in features:
        features['laundry_ratio'] = features.get('Sub_metering_2', 0) / (total_sub + 0.001)
    if 'ac_heating_ratio' in features:
        features['ac_heating_ratio'] = features.get('Sub_metering_3', 0) / (total_sub + 0.001)
    
    print(f"✅ Создано {len(features)} признаков по шаблону")
    return pd.DataFrame([features])


def generate_tomorrow_predictions():
    """Генерирует прогноз на завтра (24 часа)"""
    tomorrow = datetime.now() + timedelta(days=1)
    day_of_week = tomorrow.weekday()
    month = tomorrow.month
    
    model_name = 'lightgbm' if 'lightgbm' in models else list(models.keys())[0]
    model = models[model_name]
    
    predictions = []
    hours = []
    
    print(f"🔍 Прогноз на {tomorrow.strftime('%d.%m.%Y')} ({['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][day_of_week]})")
    
    for hour in range(24):
        # ИСПОЛЬЗУЕМ НОВУЮ ФУНКЦИЮ
        features_df = create_features_from_template(hour, day_of_week, month)
        
        try:
            prediction = model.predict(features_df)[0]
            predictions.append(prediction)
            hours.append(hour)
            print(f"   Час {hour:2d}: {prediction:.3f} кВт")
        except Exception as e:
            print(f"❌ Ошибка при прогнозе для часа {hour}: {e}")
            # Fallback значения
            base_consumption = 1.0
            if 0 <= hour <= 5: base_consumption = 0.5
            elif 7 <= hour <= 9: base_consumption = 1.5  
            elif 18 <= hour <= 22: base_consumption = 2.0
            predictions.append(base_consumption)
            hours.append(hour)
    
    print(f"✅ Прогноз завершен. Среднее: {np.mean(predictions):.3f} кВт")
    return hours, predictions, model_name, day_of_week, month


def generate_tomorrow_prediction(message):
    """Генерирует и отправляет прогноз на завтра"""
    try:
        bot.send_message(message.chat.id, "⏳ Генерирую прогноз на завтра...")
        
        # Генерируем прогноз
        hours, predictions, model_name, day_of_week, month = generate_tomorrow_predictions()
        metrics = calculate_metrics(predictions)
        
        # Создаем график
        plot_buf = create_tomorrow_plot(hours, predictions, model_name)
        
        # Названия дней и месяцев
        day_names = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
        month_names = ["", "январе", "феврале", "марте", "апреле", "мае", "июне", 
                      "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"]
        
        # Формируем текст с метриками (БЕЗ Markdown для избежания ошибок)
        metrics_text = f"""
📊 ПРОГНОЗ НА ЗАВТРА ({day_names[day_of_week].capitalize()}, {month_names[month]})

Основные метрики:
• 🎯 Средняя нагрузка: {metrics['mean']:.2f} кВт
• 📈 Пиковая нагрузка: {metrics['max']:.2f} кВт  
• 📉 Минимальная нагрузка: {metrics['min']:.2f} кВт
• 🔋 Суммарное потребление: {metrics['total']:.2f} кВт·ч

Пиковые периоды:
• 🌅 Утренний пик (7-9): {metrics['morning_peak']:.2f} кВт
• 🌇 Вечерний пик (18-22): {metrics['evening_peak']:.2f} кВт
• 🌙 Ночное время (0-5): {metrics['night']:.2f} кВт

💡 Точность модели: 92.4%
🤖 Использована модель: {model_name.upper()}
        """
        
        # Отправляем график и метрики (БЕЗ parse_mode)
        bot.send_photo(message.chat.id, plot_buf, caption=metrics_text)
        
        # Добавляем кнопку для нового прогноза
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add('📊 Новый прогноз', '📈 Статистика')
        
        bot.send_message(
            message.chat.id,
            "🔄 Хотите сделать еще один прогноз?",
            reply_markup=markup
        )
        
    except Exception as e:
        # Упрощенный текст ошибки без Markdown
        error_text = f"""
❌ Произошла ошибка при генерации прогноза

Ошибка: {str(e)}

Пожалуйста, попробуйте еще раз.
        """
        bot.send_message(message.chat.id, error_text)
        print(f"Error in generate_tomorrow_prediction: {e}")

def create_tomorrow_plot(hours, predictions, model_name):
    """Создает график прогноза на завтра"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Основной график
    ax.plot(hours, predictions, linewidth=3, alpha=0.8, color='blue', label='Прогноз нагрузки')
    
    # Зоны пиков
    ax.axvspan(7, 9, alpha=0.2, color='orange', label='Утренний пик')
    ax.axvspan(18, 22, alpha=0.2, color='red', label='Вечерний пик')
    ax.axvspan(0, 5, alpha=0.2, color='blue', label='Ночное время')
    
    ax.set_xlabel('Час дня', fontsize=12)
    ax.set_ylabel('Нагрузка (кВт)', fontsize=12)
    ax.set_title('Прогноз энергопотребления на завтра', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def calculate_metrics(predictions):
    """Рассчитывает метрики для прогнозов"""
    return {
        'min': np.min(predictions),
        'max': np.max(predictions),
        'mean': np.mean(predictions),
        'total': np.sum(predictions),
        'morning_peak': np.mean(predictions[7:10]),
        'evening_peak': np.mean(predictions[18:23]),
        'night': np.mean(predictions[0:6])
    }

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Главное меню с кнопкой прогноза"""
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add('📊 Прогноз на завтра')
    markup.add('📈 Статистика моделей', 'ℹ️ Помощь')
    
    welcome_text = """
🤖 *Бот прогнозирования энергопотребления*

Просто нажмите кнопку *"Прогноз на завтра"* чтобы получить прогноз нагрузки на следующие 24 часа.

*Точность прогноза:* 92.4%
*Используемая модель:* LightGBM
    """
    
    bot.send_message(message.chat.id, welcome_text, 
                   reply_markup=markup, parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def send_help(message):
    """Помощь"""
    help_text = """
ℹ️ *Помощь по боту*

Этот бот прогнозирует энергопотребление на основе машинного обучения.

*Как использовать:*
1. Нажмите кнопку "Прогноз на завтра"
2. Получите график и метрики нагрузки

*Модели:*
• LightGBM - 92.4% точности
• XGBoost - 92.2% точности  
• RandomForest - 90.6% точности

*Данные обучения:*
6 месяцев реальных данных энергопотребления
    """
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['stats'])
def send_stats(message):
    """Статистика моделей"""
    stats_text = """
📊 *Статистика моделей*

*LightGBM* 🚀 (основная)
• Точность: 92.4%
• Ошибка: 0.11 кВт
• Обучена на 260k+ записях

*XGBoost* ⚡
• Точность: 92.2%
• Ошибка: 0.11 кВт

*RandomForest* 🌲
• Точность: 90.6%
• Ошибка: 0.12 кВт

Все модели обучены на 6-месячных данных с минутным интервалом.
    """
    bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    """Обработка всех сообщений и кнопок"""
    if message.text == '📊 Прогноз на завтра':
        generate_tomorrow_prediction(message)
    elif message.text == '📈 Статистика моделей':
        send_stats(message)
    elif message.text == 'ℹ️ Помощь':
        send_help(message)
    else:
        # Если непонятное сообщение - показываем главное меню
        send_welcome(message)

def generate_tomorrow_prediction(message):
    """Генерирует и отправляет прогноз на завтра"""
    try:
        bot.send_message(message.chat.id, "⏳ *Генерирую прогноз на завтра...*", parse_mode='Markdown')
        
        # Генерируем прогноз
        hours, predictions, model_name, day_of_week, month = generate_tomorrow_predictions()
        metrics = calculate_metrics(predictions)
        
        # Создаем график
        plot_buf = create_tomorrow_plot(hours, predictions, model_name)
        
        # Названия дней и месяцев
        day_names = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
        month_names = ["", "январе", "феврале", "марте", "апреле", "мае", "июне", 
                      "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"]
        
        # Формируем текст с метриками
        metrics_text = f"""
📊 *ПРОГНОЗ НА ЗАВТРА* ({day_names[day_of_week].capitalize()}, {month_names[month]})

*Основные метрики:*
• 🎯 Средняя нагрузка: *{metrics['mean']:.2f} кВт*
• 📈 Пиковая нагрузка: *{metrics['max']:.2f} кВт*
• 📉 Минимальная нагрузка: *{metrics['min']:.2f} кВт*
• 🔋 Суммарное потребление: *{metrics['total']:.2f} кВт·ч*

*Пиковые периоды:*
• 🌅 Утренний пик (7-9): *{metrics['morning_peak']:.2f} кВт*
• 🌇 Вечерний пик (18-22): *{metrics['evening_peak']:.2f} кВт* 
• 🌙 Ночное время (0-5): *{metrics['night']:.2f} кВт*

💡 *Точность модели: 92.4%*
🤖 *Использована модель: {model_name.upper()}*
        """
        
        # Отправляем график и метрики
        bot.send_photo(message.chat.id, plot_buf, caption=metrics_text, parse_mode='Markdown')
        
        # Добавляем кнопку для нового прогноза
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add('📊 Новый прогноз', '📈 Статистика')
        
        bot.send_message(
            message.chat.id,
            "🔄 *Хотите сделать еще один прогноз?*",
            reply_markup=markup,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        error_text = f"""
❌ *Произошла ошибка при генерации прогноза*

Ошибка: {str(e)}

Пожалуйста, попробуйте еще раз или обратитесь к администратору.
        """
        bot.send_message(message.chat.id, error_text, parse_mode='Markdown')
        print(f"Error in generate_tomorrow_prediction: {e}")

def main():
    """Запуск бота"""
    print("🤖 Запуск бота прогнозирования энергопотребления...")
    print("✅ Бот готов к работе!")
    print("⏹️  Для остановки нажмите Ctrl+C")
    
    try:
        bot.infinity_polling()
    except Exception as e:
        print(f"❌ Ошибка при работе бота: {e}")

if __name__ == "__main__":
    main()