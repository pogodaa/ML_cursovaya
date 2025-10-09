import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== ПЕРЕОБУЧЕНИЕ МОДЕЛИ БЕЗ УТЕЧЕК ===")

# Загрузи данные
df = pd.read_csv('df/obr.csv', parse_dates=['datetime'], index_col='datetime')
print(f"✅ Данные загружены: {len(df)} записей")

# БЕЗОПАСНЫЕ ПРИЗНАКИ (без утечек)
safe_features = [
    # Циклические
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
    'day_of_week_sin', 'day_of_week_cos',
    
    # Базовые временные
    'hour', 'day_of_week', 'month', 'is_weekend',
    
    # Суточные паттерны из EDA
    'is_evening_peak', 'is_morning_peak', 'is_night', 'is_midday',
    'is_early_morning', 'is_late_evening', 'is_deep_night',
    
    # Недельные паттерны из EDA  
    'is_monday', 'is_friday', 'is_sunday', 'is_week_start', 'is_week_end',
    'weekend_evening_boost', 'weekend_morning',
    
    # Сезонные паттерны из EDA
    'is_high_season', 'is_low_season', 'is_spring',
    
    # Критические переходы из EDA
    'morning_surge_6_7', 'evening_surge_17_18', 'evening_drop_22_23',
    
    # Взаимодействия из EDA
    'winter_evening', 'summer_afternoon', 'workday_evening', 'sunday_evening',
    
    # Суб-счетчики
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'kitchen_active', 'laundry_active', 'ac_heating_active'
]

# Создаем циклические признаки
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Создаем паттерны из EDA
df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
df['is_early_morning'] = ((df['hour'] >= 4) & (df['hour'] <= 6)).astype(int)
df['is_late_evening'] = ((df['hour'] >= 21) & (df['hour'] <= 23)).astype(int)
df['is_deep_night'] = ((df['hour'] >= 1) & (df['hour'] <= 4)).astype(int)

df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_monday'] = (df['day_of_week'] == 0).astype(int)
df['is_friday'] = (df['day_of_week'] == 4).astype(int)
df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
df['is_week_start'] = (df['day_of_week'].isin([0, 1])).astype(int)
df['is_week_end'] = (df['day_of_week'].isin([4, 5])).astype(int)

df['weekend_evening_boost'] = ((df['is_weekend'] == 1) & (df['is_evening_peak'] == 1)).astype(int)
df['weekend_morning'] = ((df['is_weekend'] == 1) & (df['is_morning_peak'] == 1)).astype(int)

df['is_high_season'] = df['month'].isin([12, 1, 2]).astype(int)
df['is_low_season'] = df['month'].isin([6, 7, 8]).astype(int)
df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)

df['morning_surge_6_7'] = ((df['hour'] >= 6) & (df['hour'] <= 7)).astype(int)
df['evening_surge_17_18'] = ((df['hour'] >= 17) & (df['hour'] <= 18)).astype(int)
df['evening_drop_22_23'] = ((df['hour'] >= 22) & (df['hour'] <= 23)).astype(int)

df['winter_evening'] = ((df['is_high_season'] == 1) & (df['is_evening_peak'] == 1)).astype(int)
df['summer_afternoon'] = ((df['is_low_season'] == 1) & (df['is_midday'] == 1)).astype(int)
df['workday_evening'] = ((df['is_weekend'] == 0) & (df['is_evening_peak'] == 1)).astype(int)
df['sunday_evening'] = ((df['day_of_week'] == 6) & (df['is_evening_peak'] == 1)).astype(int)

# Активность суб-счетчиков
df['kitchen_active'] = (df['Sub_metering_1'] > 0).astype(int)
df['laundry_active'] = (df['Sub_metering_2'] > 0).astype(int)
df['ac_heating_active'] = (df['Sub_metering_3'] > 0).astype(int)

print(f"✅ Создано {len(safe_features)} безопасных признаков")

# Подготовка данных
X = df[safe_features]
y = df['Global_active_power']

# Временное разделение
split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"📊 Размеры: train {X_train.shape}, test {X_test.shape}")

# Обучаем модель
print("🎯 Обучение LightGBM...")
model = LGBMRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train)

# Оценка
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Train MAE: {train_mae:.4f} кВт, R²: {train_r2:.4f}")
print(f"Test  MAE: {test_mae:.4f} кВт, R²: {test_r2:.4f}")

# Сохраняем модель
joblib.dump(model, 'models/safe_model_lightgbm.pkl')

# Сохраняем имена признаков
import json
with open('models/safe_feature_names.json', 'w') as f:
    json.dump(safe_features, f)

print(f"💾 Модель сохранена: safe_model_lightgbm.pkl")
print(f"📋 Признаки сохранены: safe_feature_names.json ({len(safe_features)} признаков)")

# Тест на нескольких примерах
print(f"\n=== ТЕСТ ПРЕДСКАЗАНИЙ ===")
sample_predictions = model.predict(X_test.head(5))
print("Примеры (реальное → предсказание):")
for i, (real, pred) in enumerate(zip(y_test.head(5), sample_predictions)):
    print(f"  {real:.3f} → {pred:.3f} кВт")

print(f"\n🎯 ГОТОВО! Модель без утечек обучена!")