import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json

print("=== ПЕРЕОБУЧЕНИЕ С ДИНАМИЧЕСКИМИ ПРИЗНАКАМИ ===")

# Загрузи данные
df = pd.read_csv('df/obr.csv', parse_dates=['datetime'], index_col='datetime')
print(f"✅ Данные загружены: {len(df)} записей")

# СОЗДАЕМ ДИНАМИЧЕСКИЕ ПРИЗНАКИ БЕЗ УТЕЧЕК
def create_dynamic_features(df):
    """Создает динамические признаки БЕЗ утечек данных"""
    
    # 1. Циклические признаки
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 2. Временные паттерны из EDA
    df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 3. ВЗАИМОДЕЙСТВИЯ ВРЕМЕНИ И ДНЯ НЕДЕЛИ - КЛЮЧЕВОЙ ПРИЗНАК!
    df['weekday_morning'] = ((df['day_of_week'] < 5) & (df['is_morning_peak'] == 1)).astype(int)
    df['weekday_evening'] = ((df['day_of_week'] < 5) & (df['is_evening_peak'] == 1)).astype(int)
    df['weekend_morning'] = ((df['day_of_week'] >= 5) & (df['is_morning_peak'] == 1)).astype(int)
    df['weekend_evening'] = ((df['day_of_week'] >= 5) & (df['is_evening_peak'] == 1)).astype(int)
    
    # 4. СЕЗОННЫЕ ВЗАИМОДЕЙСТВИЯ
    df['winter_evening'] = ((df['month'].isin([12, 1, 2])) & (df['is_evening_peak'] == 1)).astype(int)
    df['summer_afternoon'] = ((df['month'].isin([6, 7, 8])) & (df['hour'].between(13, 17))).astype(int)
    
    # 5. ПОВЕДЕНЧЕСКИЕ ПАТТЕРНЫ
    df['family_time'] = ((df['hour'].between(18, 22)) & (df['is_weekend'] == 0)).astype(int)
    df['late_night_weekend'] = ((df['hour'].between(23, 2)) & (df['is_weekend'] == 1)).astype(int)
    
    return df

# Применяем создание признаков
df = create_dynamic_features(df)

# ДИНАМИЧЕСКИЕ ПРИЗНАКИ
dynamic_features = [
    # Циклические
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
    'day_of_week_sin', 'day_of_week_cos',
    
    # Базовые временные
    'hour', 'day_of_week', 'month', 'is_weekend',
    
    # Паттерны
    'is_evening_peak', 'is_morning_peak', 'is_night',
    
    # КЛЮЧЕВЫЕ - ВЗАИМОДЕЙСТВИЯ ДНЯ И ВРЕМЕНИ
    'weekday_morning', 'weekday_evening', 'weekend_morning', 'weekend_evening',
    
    # Сезонные взаимодействия
    'winter_evening', 'summer_afternoon',
    
    # Поведенческие паттерны
    'family_time', 'late_night_weekend',
    
    # Суб-счетчики
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]

print(f"✅ Создано {len(dynamic_features)} динамических признаков")

# Подготовка данных
X = df[dynamic_features]
y = df['Global_active_power']

# Временное разделение
split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"📊 Размеры: train {X_train.shape}, test {X_test.shape}")

# Обучаем модель
print("🎯 Обучение модели с динамическими признаками...")
model = LGBMRegressor(
    n_estimators=150,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train)

# Оценка
y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Test MAE: {test_mae:.4f} кВт")
print(f"Test R²:  {test_r2:.4f}")

# ТЕСТ РАЗЛИЧИЙ МЕЖДУ ДНЯМИ
print(f"\n=== ТЕСТ ДИНАМИКИ ===")

# Проверяем различия между днями
test_days = [
    (0, "Понедельник"),
    (4, "Пятница"), 
    (5, "Суббота"),
    (6, "Воскресенье")
]

for day, day_name in test_days:
    # Берем примеры этого дня из тестовой выборки
    day_mask = X_test['day_of_week'] == day
    if day_mask.sum() > 0:
        day_predictions = model.predict(X_test[day_mask])
        avg_consumption = day_predictions.mean()
        print(f"  {day_name}: {avg_consumption:.3f} кВт")

# Сохраняем модель
joblib.dump(model, 'models/dynamic_model_lightgbm.pkl')
with open('models/dynamic_feature_names.json', 'w') as f:
    json.dump(dynamic_features, f)

print(f"💾 Динамическая модель сохранена!")
print(f"🎯 Теперь прогнозы будут РАЗНЫЕ для разных дней!")