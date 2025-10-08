import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Конфигурация бота
BOT_TOKEN = os.getenv('BOT_TOKEN')

# Проверяем, что токен загрузился
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env файле!")

# АБСОЛЮТНЫЙ путь к моделям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.json')
MODEL_PATHS = {
    'lightgbm': os.path.join(BASE_DIR, 'models', 'lightgbm_model.pkl'),
    'xgboost': os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'), 
    'randomforest': os.path.join(BASE_DIR, 'models', 'randomforest_model.pkl')
}

# Настройки предсказания
DEFAULT_VALUES = {
    'hour': 12,
    'day_of_week': 2,  # Среда
    'month': 5,        # Май
    'Sub_metering_1': 0.0,
    'Sub_metering_2': 0.0, 
    'Sub_metering_3': 0.0,
    'Global_reactive_power': 0.1,
    'Voltage': 240.0,
    'Global_intensity': 2.5
}

# Настройки бота
BOT_CONFIG = {
    'parse_mode': 'Markdown',
    'timeout': 30
}

# Функция для проверки конфигурации
def check_config():
    """Проверяет что все настройки загружены корректно"""
    print(f"🔧 Конфигурация загружена. Токен: {BOT_TOKEN[:15]}...")
    
    available_models = []
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            print(f"✅ {model_name}: найдена")
            available_models.append(model_name)
        else:
            print(f"❌ {model_name}: не найдена ({model_path})")
    
    return available_models

print("Конфигурация загружена. Токен:", BOT_TOKEN[:10] + "..." if BOT_TOKEN else "НЕ НАЙДЕН")