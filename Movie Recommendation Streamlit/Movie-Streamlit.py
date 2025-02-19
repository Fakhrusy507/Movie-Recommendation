import streamlit as st
import pandas as pd
from data.load_data import load_data
from models.user_based import create_user_similarity_matrix, recommend_user_based
from models.item_based import create_item_similarity_matrix, recommend_item_based
from models.svd_model import train_svd_model, recommend_svd

# Judul Aplikasi
st.title("🎬 Sistem Rekomendasi Film")

# Deskripsi Aplikasi
st.markdown("""
Sistem ini memberikan rekomendasi film berdasarkan:
- **User-Based Collaborative Filtering**: Rekomendasi berdasarkan pengguna dengan preferensi serupa.
- **Item-Based Collaborative Filtering**: Rekomendasi berdasarkan film yang mirip.
- **Model-Based Collaborative Filtering (SVD)**: Rekomendasi menggunakan metode SVD.
""")

# Load Data
df, movies = load_data()

# Buat Matriks Kesamaan
user_item_matrix, user_similarity = create_user_similarity_matrix(df)
item_user_matrix, item_similarity = create_item_similarity_matrix(user_item_matrix)
svd_model = train_svd_model(df)

# Sidebar untuk Memilih Metode Rekomendasi
st.sidebar.title("Pilih Metode Rekomendasi")
method = st.sidebar.radio(
    "Metode:",
    ["User-Based", "Item-Based", "Model-Based (SVD)"]
)

# Tampilkan Rekomendasi Berdasarkan Metode yang Dipilih
if method == "User-Based":
    st.header("User-Based Collaborative Filtering")
    user_id = st.number_input("Masukkan User ID:", min_value=1, max_value=943, value=1)
    if st.button("Rekomendasi"):
        recommendations = recommend_user_based(user_id, user_item_matrix, user_similarity, df, movies)
        st.subheader("Rekomendasi Film:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

elif method == "Item-Based":
    st.header("Item-Based Collaborative Filtering")
    movie_id = st.number_input("Masukkan Movie ID:", min_value=1, max_value=1682, value=1)
    if st.button("Rekomendasi"):
        recommendations = recommend_item_based(movie_id, item_user_matrix, item_similarity, movies)
        st.subheader("Rekomendasi Film:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

elif method == "Model-Based (SVD)":
    st.header("Model-Based Collaborative Filtering (SVD)")
    user_id = st.number_input("Masukkan User ID:", min_value=1, max_value=943, value=1)
    if st.button("Rekomendasi"):
        recommendations = recommend_svd(user_id, svd_model, df, movies)
        st.subheader("Rekomendasi Film:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

# Tampilkan Data Film
st.sidebar.markdown("---")
if st.sidebar.checkbox("Tampilkan Data Film"):
    st.subheader("Data Film")
    st.write(movies)