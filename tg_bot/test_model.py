import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import mean_absolute_error

print("=== ТЕСТ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ ===\n")

# Загрузи модель
model = joblib.load('models/best_model_lightgbm_clean.pkl')
with open('models/feature_names_clean.json', 'r') as f:
    feature_names = json.load(f)

print(f"✅ Модель загружена, ожидает {len(feature_names)} признаков")

# Загрузи исходные данные
df = pd.read_csv('df/obr.csv', parse_dates=['datetime'], index_col='datetime')
print(f"✅ Данные загружены: {len(df)} записей")

# НУЖНО СОЗДАТЬ ПРИЗНАКИ КАК ПРИ ОБУЧЕНИИ!
print("Создаю признаки...")

# 1. ЦИКЛИЧЕСКИЕ ПРИЗНАКИ
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# 2. БАЗОВЫЕ ПАТТЕРНЫ (добавь самые важные)
df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print(f"✅ Создано {len([col for col in feature_names if col in df.columns])} признаков из {len(feature_names)}")

# Проверим какие признаки есть, а каких нет
missing_features = [f for f in feature_names if f not in df.columns]
print(f"❌ Отсутствует {len(missing_features)} признаков: {missing_features[:5]}...")

# Возьмем только те признаки, которые есть
available_features = [f for f in feature_names if f in df.columns]
print(f"🔄 Буду использовать {len(available_features)} доступных признаков")

# Тестируем на последних 1000 записях
X_test = df[available_features].iloc[-1000:]
y_test = df['Global_active_power'].iloc[-1000:]

print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТА ===")

# Предсказания
predictions = model.predict(X_test)

print(f"Реальные значения: {y_test.mean():.3f} ± {y_test.std():.3f} кВт")
print(f"Предсказания:      {predictions.mean():.3f} ± {predictions.std():.3f} кВт")
print(f"MAE: {mean_absolute_error(y_test, predictions):.3f} кВт")

# Проверь диапазон
print(f"\nДиапазон реальных:    {y_test.min():.3f} - {y_test.max():.3f} кВт")
print(f"Диапазон предсказаний: {predictions.min():.3f} - {predictions.max():.3f} кВт")

# Проверь несколько примеров
print(f"\nПримеры (реальное → предсказание):")
for i in range(5):
    print(f"  {y_test.iloc[i]:.3f} → {predictions[i]:.3f} кВт")

# Анализ проблемы
print(f"\n=== АНАЛИЗ ===")
if abs(predictions.mean() - y_test.mean()) > 1.0:
    print("❌ СЕРЬЕЗНАЯ ПРОБЛЕМА: Модель предсказывает нереалистичные значения")
    print("   Причина: Отсутствуют ключевые признаки для корректного предсказания")
else:
    print("✅ Модель работает нормально")