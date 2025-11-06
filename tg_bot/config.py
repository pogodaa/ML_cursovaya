# config.py - УПРОЩЕННАЯ ВЕРСИЯ ДЛЯ SARIMA
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env файле!")

# Пути к моделям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sarima_best_model.pkl')

print("✅ Конфигурация SARIMA загружена")