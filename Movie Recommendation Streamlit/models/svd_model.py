from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_svd_model(df):
    """
    Melatih model SVD.

    Args:
        df (pd.DataFrame): DataFrame yang berisi rating dan informasi film.

    Returns:
        model (SVD): Model SVD yang telah dilatih.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    print("Model-Based (SVD) - RMSE:", accuracy.rmse(predictions))
    return model

def recommend_svd(user_id, model, df, movies, top_n=5):
    """
    Memberikan rekomendasi film menggunakan model SVD.

    Args:
        user_id (int): ID pengguna.
        model (SVD): Model SVD yang telah dilatih.
        df (pd.DataFrame): DataFrame yang berisi rating dan informasi film.
        movies (pd.DataFrame): DataFrame yang berisi informasi film.
        top_n (int): Jumlah rekomendasi yang akan diberikan.

    Returns:
        list: Daftar film yang direkomendasikan.
    """
    all_movie_ids = df['movie_id'].unique()
    user_rated_movies = df[df['user_id'] == user_id]['movie_id'].tolist()
    unrated_movies = [m for m in all_movie_ids if m not in user_rated_movies]

    predictions = [(m, model.predict(user_id, m).est) for m in unrated_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return movies[movies['movie_id'].isin([m[0] for m in predictions[:top_n]])]['title'].tolist()