# bot.py - УПРОЩЕННЫЙ БОТ ДЛЯ SARIMA
import telebot
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Загружаем токен
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    print("❌ BOT_TOKEN не найден!")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# Загружаем SARIMA модель
try:
    model = joblib.load('models/sarima_best_model.pkl')
    print("✅ SARIMA модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

def predict_sarima(days_ahead=1):
    """Прогноз с SARIMA моделью"""
    try:
        # SARIMA прогнозирует на заданное количество шагов вперед
        forecast = model.forecast(steps=days_ahead)
        
        # Для простоты возвращаем дневной прогноз
        prediction = float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') else float(forecast[-1])
        
        # Защита от отрицательных значений
        prediction = max(0.1, prediction)
        
        return round(prediction, 2)
        
    except Exception as e:
        print(f"❌ Ошибка прогноза: {e}")
        return 1.0  # Значение по умолчанию

def create_main_keyboard():
    """Создает основную клавиатуру"""
    keyboard = InlineKeyboardMarkup()
    keyboard.row(
        InlineKeyboardButton("📊 Прогноз на сегодня", callback_data="forecast_today"),
        InlineKeyboardButton("📈 Прогноз на завтра", callback_data="forecast_tomorrow")
    )
    keyboard.row(
        InlineKeyboardButton("📅 Прогноз на неделю", callback_data="forecast_week")
    )
    return keyboard

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
⚡ *Бот прогнозирования суточной нагрузки в электросети*

*Тема:* Прогнозирование на основе анализа временных рядов и сезонных факторов

*Модель:* SARIMA с учетом недельной сезонности
*Точность:* 68.7% (MAPE 31.3%)

*Выберите опцию:*
• **Прогноз на сегодня** - средняя нагрузка за сутки
• **Прогноз на завтра** - средняя нагрузка за сутки  
• **Прогноз на неделю** - детальный график на 7 дней

*Используйте кнопки ниже для получения прогнозов:*
    """
    bot.send_message(message.chat.id, welcome_text, 
                   parse_mode='Markdown',
                   reply_markup=create_main_keyboard())

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    try:
        if call.data == "forecast_today":
            bot.answer_callback_query(call.id, "Рассчитываю нагрузку на сегодня...")
            send_today_forecast(call.message)
            
        elif call.data == "forecast_tomorrow":
            bot.answer_callback_query(call.id, "Рассчитываю нагрузку на завтра...")
            send_tomorrow_forecast(call.message)
            
        elif call.data == "forecast_week":
            bot.answer_callback_query(call.id, "Строю недельный прогноз...")
            send_weekly_forecast(call.message)
            
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Ошибка: {str(e)}")

def send_today_forecast(message):
    """Просто число - прогноз на сегодня"""
    try:
        prediction = predict_sarima(1)
        today = datetime.now().strftime('%d.%m.%Y')
        
        # Определяем уровень нагрузки
        if prediction > 1.5:
            level = "🔴 ВЫСОКАЯ"
            advice = "Рекомендуется снизить энергоемкие процессы"
        elif prediction > 1.0:
            level = "🟡 СРЕДНЯЯ" 
            advice = "Нормальный режим работы"
        else:
            level = "🟢 НИЗКАЯ"
            advice = "Благоприятный период для энергоемких задач"
        
        response = f"""
📊 *ПРОГНОЗ НА СЕГОДНЯ* ({today})

⚡ *Средняя суточная нагрузка:* `{prediction} кВт`

📈 *Уровень нагрузки:* {level}
💡 *Рекомендация:* {advice}

*Метрика точности:* 68.7%
        """
        
        bot.send_message(message.chat.id, response, 
                       parse_mode='Markdown',
                       reply_markup=create_main_keyboard())
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка: {str(e)}")

def send_tomorrow_forecast(message):
    """Просто число - прогноз на завтра"""
    try:
        prediction = predict_sarima(2)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%d.%m.%Y')
        
        # Определяем уровень нагрузки
        if prediction > 1.5:
            level = "🔴 ВЫСОКАЯ"
            advice = "Запланируйте энергоемкие работы на другое время"
        elif prediction > 1.0:
            level = "🟡 СРЕДНЯЯ" 
            advice = "Стандартный режим планирования"
        else:
            level = "🟢 НИЗКАЯ"
            advice = "Идеальный день для энергоемких процессов"
        
        response = f"""
📈 *ПРОГНОЗ НА ЗАВТРА* ({tomorrow})

⚡ *Средняя суточная нагрузка:* `{prediction} кВт`

📊 *Уровень нагрузки:* {level}
🎯 *Планирование:* {advice}

*Метрика точности:* 68.7%
        """
        
        bot.send_message(message.chat.id, response, 
                       parse_mode='Markdown',
                       reply_markup=create_main_keyboard())
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка: {str(e)}")

def send_weekly_forecast(message):
    """Полный график на неделю"""
    try:
        # Прогноз на 7 дней
        days = range(1, 8)
        predictions = [predict_sarima(i) for i in days]
        dates = [(datetime.now() + timedelta(days=i)).strftime('%d.%m') for i in days]
        day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        
        # Создаем красивый график
        plt.figure(figsize=(14, 8))
        
        # График с заливкой
        plt.fill_between(range(7), predictions, alpha=0.3, color='skyblue')
        plt.plot(range(7), predictions, 'bo-', linewidth=3, markersize=8, markerfacecolor='red')
        
        # Подписи
        for i, (date, pred, day_name) in enumerate(zip(dates, predictions, day_names)):
            plt.annotate(f'{pred} кВт', (i, pred), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
            plt.annotate(f'{date}\n{day_name}', (i, 0), textcoords="offset points", 
                        xytext=(0,-25), ha='center', fontsize=8)
        
        plt.title('📅 ПРОГНОЗ СУТОЧНОЙ НАГРУЗКИ НА НЕДЕЛЮ\n', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Средняя нагрузка (кВт)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(7), [''] * 7)  # Убираем стандартные подписи
        plt.ylim(0, max(predictions) * 1.3)
        
        # Добавляем линии уровней
        avg_load = np.mean(predictions)
        plt.axhline(y=avg_load, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Среднее: {avg_load:.2f} кВт')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Статистика
        max_day = dates[np.argmax(predictions)]
        max_value = max(predictions)
        min_day = dates[np.argmin(predictions)]
        min_value = min(predictions)
        
        caption = f"""
📈 *НЕДЕЛЬНЫЙ ПРОГНОЗ НАГРУЗКИ*

*Статистика:*
• 🟠 Средняя нагрузка: `{avg_load:.2f} кВт`
• 🔴 Максимум: `{max_value} кВт` ({max_day})
• 🟢 Минимум: `{min_value} кВт` ({min_day})
• 📊 Размах: `{max_value - min_value:.2f} кВт`

*Анализ сезонности:*
✓ Учтены недельные паттерны
✓ Учтены исторические тренды  
✓ Прогноз на основе SARIMA модели
        """
        
        bot.send_photo(message.chat.id, buf, caption=caption, 
                      parse_mode='Markdown',
                      reply_markup=create_main_keyboard())
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка построения графика: {str(e)}")

@bot.message_handler(commands=['predict'])
def send_predict_menu(message):
    """Меню прогнозов"""
    menu_text = """
⚡ *Прогнозирование суточной нагрузки в электросети*

*Выберите период прогноза:*

• **Прогноз на сегодня** - текущая суточная нагрузка
• **Прогноз на завтра** - планирование на следующий день  
• **Прогноз на неделю** - детальный анализ на 7 дней

*Модель SARIMA учитывает:*
✓ Временные ряды потребления
✓ Недельную сезонность
✓ Исторические паттерны нагрузки
    """
    bot.send_message(message.chat.id, menu_text, 
                   parse_mode='Markdown',
                   reply_markup=create_main_keyboard())

@bot.message_handler(commands=['info'])
def send_info(message):
    """Информация о модели"""
    info_text = """
🔬 *Информация о системе прогнозирования*

*Тема исследования:*
"Прогнозирование суточной нагрузки в электросети на основе анализа временных рядов и сезонных факторов"

*Используемая модель:*
• SARIMA(1,1,1)(1,1,1,7)
• Учет недельной сезонности
• Обучена на реальных данных энергопотребления

*Метрики качества:*
• MAE: 0.244 кВт
• MAPE: 31.3%
• Учет сезонных факторов: ✅

*Научная ценность:*
Система демонстрирует применение методов анализа временных рядов для решения практических задач энергетики.
    """
    bot.send_message(message.chat.id, info_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    """Обработка любого текстового сообщения"""
    help_text = """
⚡ *Бот прогнозирования энергопотребления*

Для получения прогнозов используйте:
• Кнопки ниже
• Команду /predict
• Команду /info для информации о модели

*Тема:* Прогнозирование суточной нагрузки на основе временных рядов
    """
    bot.send_message(message.chat.id, help_text,
                   parse_mode='Markdown',
                   reply_markup=create_main_keyboard())

if __name__ == "__main__":
    print("🚀 Бот прогнозирования энергопотребления запущен!")
    print("✅ SARIMA модель загружена и готова к работе")
    print("📊 Доступны прогнозы: сегодня, завтра, неделя")
    bot.infinity_polling()