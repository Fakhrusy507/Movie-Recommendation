import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def create_user_similarity_matrix(df):
    """
    Membuat matriks kesamaan pengguna berdasarkan data rating.

    Args:
        df (pd.DataFrame): DataFrame yang berisi rating dan informasi film.

    Returns:
        user_item_matrix (pd.DataFrame): Matriks item-pengguna.
        user_similarity (np.array): Matriks kesamaan pengguna.
    """
    # Buat matriks item-pengguna (user-item matrix)
    user_item_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    
    # Konversi ke matriks sparse untuk efisiensi
    user_sparse_matrix = csr_matrix(user_item_matrix.values)
    
    # Hitung kesamaan pengguna menggunakan cosine similarity
    user_similarity = cosine_similarity(user_sparse_matrix)
    
    return user_item_matrix, user_similarity

def recommend_user_based(user_id, user_item_matrix, user_similarity, df, movies, top_n=5):
    """
    Memberikan rekomendasi film berdasarkan pengguna yang mirip.

    Args:
        user_id (int): ID pengguna.
        user_item_matrix (pd.DataFrame): Matriks item-pengguna.
        user_similarity (np.array): Matriks kesamaan pengguna.
        df (pd.DataFrame): DataFrame yang berisi rating dan informasi film.
        movies (pd.DataFrame): DataFrame yang berisi informasi film.
        top_n (int): Jumlah rekomendasi yang akan diberikan.

    Returns:
        list: Daftar film yang direkomendasikan.
    """
    if user_id not in user_item_matrix.index:
        return ["User ID tidak ditemukan."]

    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = np.argsort(-user_similarity[user_idx])[1:top_n+1]

    recommended_movies = []
    for similar_user in similar_users:
        similar_user_id = user_item_matrix.index[similar_user]
        unseen_movies = df[df['user_id'] == similar_user_id]['movie_id'].tolist()
        recommended_movies.extend(unseen_movies)

    recommended_movies = list(set(recommended_movies))[:top_n]
    return movies[movies['movie_id'].isin(recommended_movies)]['title'].tolist()