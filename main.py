import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

print(movies.head())
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head())
movies.dropna(inplace=True)
import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
movies['keywords'] = movies['keywords'].apply(lambda x: " ".join(x))
movies['cast'] = movies['cast'].apply(lambda x: " ".join(x))
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast']


movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    for i in movie_list:
        print(movies.iloc[i[0]].title)
recommend("Avatar")