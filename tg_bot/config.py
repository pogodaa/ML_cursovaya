# config.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env файле!")

# Пути к моделям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.json')
MODEL_PATHS = {
    'lightgbm': os.path.join(BASE_DIR, 'models', 'lightgbm_model.pkl')
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
    
    # Проверяем только LightGBM (остальные не используем)
    model_path = MODEL_PATHS['lightgbm']
    if os.path.exists(model_path):
        print(f"✅ LightGBM модель найдена")
        return ['lightgbm']
    else:
        print(f"❌ LightGBM модель не найдена ({model_path})")
        return []

print("Конфигурация загружена. Токен:", BOT_TOKEN[:10] + "..." if BOT_TOKEN else "НЕ НАЙДЕН")