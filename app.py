import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import gdown

# Гарчиг
st.set_page_config(page_title="Байрны үнэ таавар", layout="centered")
st.title("🏠 Улаанбаатарын орон сууцны үнийг таамаглах апп")
st.markdown("Таны оруулсан мэдээллээр байрны зах зээлийн үнийг тооцоолно (unegui.mn-ийн 15,000+ зарын өгөгдөл дээр сургагдсан)")

# Загвар болон encoder-ээ ачаалах
@st.cache_resource
def load_model():
    model_path = 'best_model.pkl'
    encoder_path = 'label_encoder.pkl'
    
    # Хэрэв файл байхгүй бол Google Drive-ээс татна
    if not os.path.exists(model_path):
        st.info("Загвар татаж байна... Түр хүлээнэ үү (эхний удаа удаан байж болно)")
        gdown.download("https://drive.google.com/file/d/11vPH3PcQbnkXF7cbvNZ1RdYaXAnd4HPI/view?usp=sharing", model_path, quiet=False)
    
    if not os.path.exists(encoder_path):
        gdown.download("https://drive.google.com/file/d/1xc0cn9JtrMGkpNElgLLQlY6giHftL7kI/view?usp=sharing", encoder_path, quiet=False)
    
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return model, le

# Хэрэглэгчийн оролт
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Талбай (м²)", min_value=10.0, max_value=500.0, value=80.0, step=1.0)
    rooms = st.slider("Өрөөний тоо", 1, 8, 3)
    floor = st.number_input("Аль давхарт вэ", min_value=1, max_value=30, value=6)
    total_floors = st.number_input("Барилгын нийт давхар", min_value=1, max_value=30, value=16)

with col2:
    year_built = st.number_input("Баригдсан он", min_value=1980, max_value=2026, value=2018)
    has_elevator = st.selectbox("Лифттэй эсэх", ["Үгүй", "Тийм"])
    has_garage = st.selectbox("Гарааштай эсэх", ["Үгүй", "Тийм"])
    windows = st.number_input("Цонхны тоо", min_value=1, max_value=10, value=4)

district = st.selectbox("Дүүрэг", sorted(le.classes_))

# Тооцоолох товч
if st.button("Үнийг тооцоолох", type="primary"):
    # Бэлтгэл
    elevator_val = 1 if has_elevator == "Тийм" else 0
    garage_val = 1 if has_garage == "Тийм" else 0
    district_encoded = le.transform([district])[0]

    # Өгөгдөл бэлтгэх
    input_data = np.array([[area, rooms, floor, total_floors, year_built,
                            elevator_val, garage_val, windows, district_encoded]])

    # Таамаглал
    prediction = model.predict(input_data)[0]

    # Хэрэв лог хувиргалттай загвар ашигласан бол буцааж хөрвүүлэх (шаардлагатай бол тайлбар хэсэгт нэмнэ үү)
    # prediction = np.expm1(prediction)

    st.markdown("---")
    st.success(f"### Таамагласан зах зээлийн үнэ: **{prediction:,.0f} ₮**")
    st.info("⚠️ Энэ бол таамаглал тул бодит борлуулалтын үнээс ±15% зөрүүтэй байж болно.")
    st.caption("Загвар: Gradient Boosting / Random Forest")

# Доод талд тайлбар
st.markdown("---")
st.caption("Зохиогч: Зоригтбаатар | VS Code + Streamlit ашиглан бүтээв")