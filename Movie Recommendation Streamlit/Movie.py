from data.load_data import load_data
from models.user_based import create_user_similarity_matrix, recommend_user_based
from models.item_based import create_item_similarity_matrix, recommend_item_based
from models.svd_model import train_svd_model, recommend_svd

def main():
    df, movies = load_data()
    user_item_matrix, user_similarity = create_user_similarity_matrix(df)
    item_user_matrix, item_similarity = create_item_similarity_matrix(user_item_matrix)
    svd_model = train_svd_model(df)

    while True:
        print("\nPilih Metode Rekomendasi:")
        print("1. User-Based Collaborative Filtering")
        print("2. Item-Based Collaborative Filtering")
        print("3. Model-Based Collaborative Filtering (SVD)")
        print("4. Keluar")
        
        choice = input("Masukkan pilihan (1/2/3/4): ")
        
        if choice == "1":
            user_id = int(input("Masukkan User ID: "))
            print("\nRekomendasi untuk User ID", user_id, ":", recommend_user_based(user_id, user_item_matrix, user_similarity, movies))
        
        elif choice == "2":
            movie_id = int(input("Masukkan Movie ID: "))
            print("\nRekomendasi film mirip dengan Movie ID", movie_id, ":", recommend_item_based(movie_id, item_user_matrix, item_similarity, movies))
        
        elif choice == "3":
            user_id = int(input("Masukkan User ID: "))
            print("\nRekomendasi untuk User ID", user_id, ":", recommend_svd(user_id, svd_model, df, movies))
        
        elif choice == "4":
            print("Keluar dari program.")
            break
        
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

if __name__ == "__main__":
    main()