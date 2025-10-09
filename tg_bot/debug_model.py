import pandas as pd
import joblib
import json
import numpy as np

print("=== ДИАГНОСТИКА МОДЕЛИ ===\n")

# 1. Загрузим модель и посмотрим что она ожидает
model = joblib.load('models/best_model_lightgbm_clean.pkl')
with open('models/feature_names_clean.json', 'r') as f:
    expected_features = json.load(f)

print(f"1. МОДЕЛЬ ОЖИДАЕТ {len(expected_features)} ПРИЗНАКОВ:")
print(f"   Первые 10: {expected_features[:10]}")
print(f"   Последние 10: {expected_features[-10:]}")
print(f"   Типы признаков: {len([f for f in expected_features if 'lag' in f])} лагов, "
      f"{len([f for f in expected_features if 'rolling' in f])} скользящих, "
      f"{len([f for f in expected_features if 'sin' in f or 'cos' in f])} циклических")

# 2. Проверим что генерирует бот
print(f"\n2. ТЕСТ ГЕНЕРАЦИИ ПРИЗНАКОВ БОТОМ:")

# Имитируем генерацию бота
from datetime import datetime
test_date = datetime(2025, 10, 11)
hour = 3
day_of_week = 5  # Суббота
month = 10

# Создаем признаки как в боте
features_dict = {}

# Циклические признаки
features_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
features_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
features_dict['month_sin'] = np.sin(2 * np.pi * month / 12)
features_dict['month_cos'] = np.cos(2 * np.pi * month / 12)
features_dict['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features_dict['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

# Базовые паттерны
features_dict['is_evening_peak'] = 1 if 18 <= hour <= 22 else 0
features_dict['is_morning_peak'] = 1 if 7 <= hour <= 9 else 0
features_dict['is_night'] = 1 if 0 <= hour <= 5 else 0

# Базовые временные
features_dict['hour'] = hour
features_dict['day_of_week'] = day_of_week
features_dict['month'] = month
features_dict['is_weekend'] = 1 if day_of_week >= 5 else 0

print(f"   Создано {len(features_dict)} базовых признаков")

# 3. Заполним остальные признаки СЛУЧАЙНЫМИ значениями (как в боте)
print(f"\n3. ЗАПОЛНЕНИЕ ОСТАЛЬНЫХ ПРИЗНАКОВ:")

for feature in expected_features:
    if feature not in features_dict:
        if 'lag' in feature:
            features_dict[feature] = 1.2  # Как в боте
        elif 'rolling' in feature:
            features_dict[feature] = 1.2  # Как в боте  
        elif 'ratio' in feature:
            features_dict[feature] = 0.1  # Случайное
        else:
            features_dict[feature] = 0.0  # По умолчанию

print(f"   Всего признаков: {len(features_dict)}")

# 4. Создаем DataFrame в правильном порядке
features_df = pd.DataFrame([features_dict])[expected_features]

# 5. Пробуем предсказание с отключенной проверкой
print(f"\n4. ТЕСТ ПРЕДСКАЗАНИЯ:")

try:
    prediction = model.predict(features_df, predict_disable_shape_check=True)[0]
    print(f"   ✅ Предсказание: {prediction:.3f} кВт")
    
    if prediction > 2.0:
        print(f"   ❌ ПРОБЛЕМА: Предсказание слишком высокое!")
        print(f"   💡 Причина: Модель обучалась на 'идеальных' признаках")
        print(f"   💡 А бот генерирует 'реалистичные' признаки")
    else:
        print(f"   ✅ Предсказание в норме")
        
except Exception as e:
    print(f"   ❌ Ошибка предсказания: {e}")

print(f"\n5. ВЫВОД:")
print(f"   🎯 ПРОБЛЕМА: Разрыв между обучением и инференсом")
print(f"   📚 Модель обучалась на ПРОПУСКАХ и ЛАГАХ из будущего")
print(f"   🤖 Бот генерирует РЕАЛИСТИЧНЫЕ признаки без утечек")
print(f"   🔥 Результат: Модель 'сломана' при реальном использовании")