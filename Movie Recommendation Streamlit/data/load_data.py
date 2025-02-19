import pandas as pd

def load_data():
    """
    Memuat dataset MovieLens 100K dari URL.

    Returns:
        df (pd.DataFrame): DataFrame yang berisi rating dan informasi film.
        movies (pd.DataFrame): DataFrame yang berisi informasi film.
    """
    ratings = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", 
                          sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    movies = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.item", 
                         sep='|', encoding='latin-1', names=['movie_id', 'title'], usecols=[0, 1])

    df = ratings.merge(movies, on="movie_id").drop(columns=['timestamp'])
    return df, movies