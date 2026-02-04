import pandas as pd
import joblib
import streamlit as st


st.sidebar.title("Informasi")
st.sidebar.success("Dibuat oleh Saskia")

st.set_page_config(
page_title = "Regresi Penjulan Tomat",
page_icon = "🍅"

)

st.title("🍅Regresi Penjualan Tomat")
st.markdown("Aplikasi machine learning regression untuk menghitung total penjualan tomat berdasarkan fitur `Harga`, `Hari`, `Cuaca`, dan `Promo`")

model_forest = joblib.load("model_forest.joblib")

harga = st.slider("Harga", 0, 20000, 7000)
hari = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
cuaca = st.selectbox("Cuaca",["Cerah", "Berawan", "Mendung", "Hujan"])
promo= st.pills("Promo", ["Ya","Tidak"], default="Tidak")

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[harga, hari, cuaca, promo]], 
							columns=["Harga","Hari", "Cuaca", "Promo" ])
	
	prediksi = model_forest.predict(data_baru)[0]
	
	st.success(f"Model memprediksi total penjualan : **{prediksi:.0f}**")
	st.balloons()
