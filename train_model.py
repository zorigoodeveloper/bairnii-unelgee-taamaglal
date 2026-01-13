# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns

print("Өгөгдөл уншиж байна...")
df = pd.read_csv('mongolia_housing.csv')

print(f"Нийт {len(df)} мөр өгөгдөл уншигдлаа.")

MIN_PRICE = 90000000  
print(f"\n9 саяас доош үнэтэй мөрүүдийг хасаж байна...")
before = len(df)
df = df[df['price'] >= MIN_PRICE].copy()
after = len(df)
print(f"Хасагдсан: {before - after} мөр ({(before - after)/before*100:.1f}%)")
print(f"Үлдсэн өгөгдөл: {after} мөр")
print(f"Шинэ хамгийн бага үнэ: {df['price'].min():,.0f} ₮")

# === ӨГӨГДӨЛ ЦЭВЭРЛЭХ ===
df['area_sqm'] = df['area'].str.replace(' м²', '').str.replace(',', '.').astype(float)

def extract_rooms(title):
    title = str(title).lower()
    match = re.search(r'(\d+)\s*өрөө', title)
    if match:
        return int(match.group(1))
    if 'студи' in title:
        return 1
    return None

df['rooms'] = df['title'].apply(extract_rooms)

def extract_district(place):
    if '—' in str(place):
        parts = str(place).split('—')
        if len(parts) >= 2:
            return parts[1].strip().split(',')[0].strip()
    return 'Тодорхойгүй'

df['district'] = df['place'].apply(extract_district)

df['has_elevator'] = df['elevator'].apply(lambda x: 1 if 'шаттай' in str(x) else 0)
df['has_garage'] = df['garage'].apply(lambda x: 1 if 'Байгаа' in str(x) else 0)

df.rename(columns={
    'price': 'price_mnt',
    'floor_number': 'floor',
    'building_floors': 'total_floors',
    'built_year': 'year_built',
    'window_count': 'windows'
}, inplace=True)

# Missing утга дүүргэх
df['area_sqm'] = df['area_sqm'].fillna(df['area_sqm'].median())
df['rooms'] = df['rooms'].fillna(df['rooms'].mode()[0] if not df['rooms'].mode().empty else 3)
df['floor'] = df['floor'].fillna(df['floor'].median())
df['year_built'] = df['year_built'].fillna(df['year_built'].median())

# District encoding
le = LabelEncoder()
df['district_encoded'] = le.fit_transform(df['district'])

# Features сонгох
features = ['area_sqm', 'rooms', 'floor', 'total_floors', 'year_built',
            'has_elevator', 'has_garage', 'windows', 'district_encoded']

X = df[features]
y = df['price_mnt']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Загваруудыг сургах
print("\nЗагваруудыг сургаж байна...")
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Үр дүн хэвлэх
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

print("\n=== ҮР ДҮН (9 саяас дээш өгөгдөл дээр) ===")
print(f"Random Forest     → R²: {r2_score(y_test, rf_pred):.4f} | MAE: {mean_absolute_error(y_test, rf_pred):,.0f} ₮")
print(f"Gradient Boosting → R²: {r2_score(y_test, gb_pred):.4f} | MAE: {mean_absolute_error(y_test, gb_pred):,.0f} ₮")

# Хамгийн сайн загварыг сонгох
best_model = rf if r2_score(y_test, rf_pred) > r2_score(y_test, gb_pred) else gb
best_name = "Random Forest" if best_model == rf else "Gradient Boosting"

print(f"\nХамгийн сайн загвар: {best_name}")

# Загвар ба encoder хадгалах (шинэчлэгдсэн!)
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n✅ Шинэ загвар хадгалагдлаа: best_model.pkl (9 саяас дээш өгөгдөл дээр сургагдсан)")
print("   label_encoder.pkl бас хадгалагдлаа.")

# Feature importance график
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title(f'Feature Importance - {best_name}')
plt.xlabel('Чухал байдал')
plt.tight_layout()
plt.show()