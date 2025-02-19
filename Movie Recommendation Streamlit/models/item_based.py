import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity  # Import fungsi cosine_similarity

def create_item_similarity_matrix(user_item_matrix):
    """
    Membuat matriks kesamaan item.

    Args:
        user_item_matrix (pd.DataFrame): Matriks item-pengguna.

    Returns:
        item_user_matrix (pd.DataFrame): Matriks item-pengguna yang sudah di-transpose.
        item_similarity (np.array): Matriks kesamaan item.
    """
    item_user_matrix = user_item_matrix.T
    item_sparse_matrix = csr_matrix(item_user_matrix.values)
    item_similarity = cosine_similarity(item_sparse_matrix)  # Gunakan fungsi cosine_similarity
    return item_user_matrix, item_similarity

def recommend_item_based(movie_id, item_user_matrix, item_similarity, movies, top_n=5):
    """
    Memberikan rekomendasi film berdasarkan item yang mirip.

    Args:
        movie_id (int): ID film.
        item_user_matrix (pd.DataFrame): Matriks item-pengguna.
        item_similarity (np.array): Matriks kesamaan item.
        movies (pd.DataFrame): DataFrame yang berisi informasi film.
        top_n (int): Jumlah rekomendasi yang akan diberikan.

    Returns:
        list: Daftar film yang direkomendasikan.
    """
    if movie_id not in item_user_matrix.index:
        return ["Movie ID tidak ditemukan."]

    movie_idx = item_user_matrix.index.get_loc(movie_id)
    similar_movies = np.argsort(-item_similarity[movie_idx])[1:top_n+1]

    return movies[movies['movie_id'].isin(item_user_matrix.index[similar_movies])]['title'].tolist()