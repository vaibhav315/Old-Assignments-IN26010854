# ==========================================================
# MOVIE RECOMMENDATION SYSTEM
# Content Based + Collaborative Filtering
# ==========================================================


# Import Libraries


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


from scipy.sparse import csr_matrix


import joblib





# ==========================================================
# 1. Load Dataset
# ==========================================================


movies = pd.read_csv(
    "movies.csv"
)


ratings = pd.read_csv(
    "ratings.csv"
)



print("Movies Dataset")

print(movies.head())


print("\nRatings Dataset")

print(ratings.head())






# ==========================================================
# 2. Dataset Information
# ==========================================================


print("\nMovies Shape:")

print(movies.shape)



print("\nRatings Shape:")

print(ratings.shape)







# ==========================================================
# 3. Exploratory Data Analysis
# ==========================================================



plt.figure(figsize=(8,5))


sns.countplot(

    x=ratings["rating"]

)


plt.title(

"Rating Distribution"

)


plt.show()






# Number of ratings per movie


movie_rating_count = ratings.groupby(

"movieId"

).size().sort_values(

ascending=False

)



print(

movie_rating_count.head()

)







# ==========================================================
# 4. Content Based Recommendation
# ==========================================================



# Combine movie information


movies["genres"] = movies["genres"].replace(

"|",

" ",

regex=True

)



# TF-IDF Vectorizer


tfidf = TfidfVectorizer(

stop_words="english"

)



tfidf_matrix = tfidf.fit_transform(

movies["genres"]

)



print(

"TF-IDF Shape:",

tfidf_matrix.shape

)





# Similarity Matrix


content_similarity = cosine_similarity(

tfidf_matrix

)





# Movie index


indices = pd.Series(

movies.index,

index=movies["title"]

).drop_duplicates()






# ==========================================================
# 5. Content Recommendation Function
# ==========================================================



def recommend_movie(movie_name, number=10):


    if movie_name not in indices:

        print(

        "Movie not found"

        )

        return



    idx = indices[movie_name]



    similarity_scores=list(

        enumerate(

            content_similarity[idx]

        )

    )



    similarity_scores=sorted(

        similarity_scores,

        key=lambda x:x[1],

        reverse=True

    )



    similarity_scores=similarity_scores[1:number+1]



    movie_indices=[

        i[0]

        for i in similarity_scores

    ]



    return movies.iloc[

        movie_indices

    ][

        [

        "title",

        "genres"

        ]

    ]







# ==========================================================
# 6. Test Content Recommendation
# ==========================================================


print(

"\nRecommended Movies:\n"

)



print(

recommend_movie(

"Toy Story (1995)"

)

)







# ==========================================================
# 7. Collaborative Filtering
# ==========================================================



# Create user movie matrix


user_movie_matrix = ratings.pivot_table(

    index="userId",

    columns="movieId",

    values="rating"

).fillna(0)





print(

user_movie_matrix.head()

)





# Convert to sparse matrix


sparse_matrix = csr_matrix(

    user_movie_matrix.values

)





# User similarity


user_similarity = cosine_similarity(

    sparse_matrix

)






# ==========================================================
# 8. User Recommendation Function
# ==========================================================



def recommend_for_user(user_id, number=10):



    user_index = user_id-1



    similarity = user_similarity[user_index]



    similar_users = list(

        enumerate(similarity)

    )


    similar_users = sorted(

        similar_users,

        key=lambda x:x[1],

        reverse=True

    )



    similar_users = similar_users[1:6]



    movie_scores={}



    for similar_user,score in similar_users:



        movies_watched = user_movie_matrix.iloc[similar_user]



        for movie_id,rating in movies_watched.items():



            if rating > 0:


                if movie_id not in movie_scores:

                    movie_scores[movie_id]=0



                movie_scores[movie_id]+=score*rating




    recommended = sorted(

        movie_scores.items(),

        key=lambda x:x[1],

        reverse=True

    )



    movie_ids=[

        x[0]

        for x in recommended[:number]

    ]



    return movies[

        movies["movieId"].isin(movie_ids)

    ][

        [

        "title",

        "genres"

        ]

    ]








# ==========================================================
# 9. Test Collaborative Recommendation
# ==========================================================


print(

"\nUser Recommendations:\n"

)


print(

recommend_for_user(

1

)

)







# ==========================================================
# 10. Save Model
# ==========================================================



model_data={

"tfidf":tfidf,

"tfidf_matrix":tfidf_matrix,

"similarity":content_similarity,

"indices":indices

}



joblib.dump(

model_data,

"recommendation_model.pkl"

)



print(

"\nRecommendation Model Saved!"

)
