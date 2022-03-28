import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from helpers import eda
pd.set_option('display.expand_frame_repr', False)


movies = eda.csv_file("movies")
ratings = eda.csv_file("ratings")
ratings=pd.merge(movies,ratings).drop(["genres","timestamp"],axis=1)

user_ratigs = ratings.pivot_table(index=["userId"],columns=["title"],values=["rating"])
user_ratigs.columns =[str(c2) for (c1,c2) in user_ratigs.columns.tolist()]
user_ratigs=user_ratigs.reset_index()
cat_cols, num_cols, cat_but_car,typless_cols= eda.col_types(user_ratigs)
na_col,null_high_col_name= eda.desc_statistics(user_ratigs, num_cols, cat_cols, refresh=True, na_rows=False, null_ratio=95)
#remove movies which have more than %95 null ratio who rated it. and fill NaN with 0
user_ratigs=user_ratigs.drop(null_high_col_name,axis=1).fillna(0)
user_ratigs.info()

#build similarity matrix
# we are taking a transpose since we want similarity between items which need to be in rows. but for users  not need transpose
item_similarity = cosine_similarity(user_ratigs.T)
item_similarity_df = pd.DataFrame(item_similarity,index=user_ratigs.columns, columns=user_ratigs.columns)
item_similarity_df=item_similarity_df[~(item_similarity_df.index=="userId")].drop("userId",axis=1)

def get_similar_movies(movie_name,user_rating):
    similar_score =item_similarity_df[movie_name]*(user_rating-2.5) # while give low rating, provide to show negative corelation with minus mean rating
    similar_score=similar_score.sort_values(ascending=False)
    return similar_score

film_lover= [("12 Angry Men (1957)",5),("101 Dalmatians (1996)",4),("10 Things I Hate About You (1999)",2)]
similar_movies=pd.DataFrame()
for movie, rating in film_lover:
    similar_movies= similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)
    if movie in similar_movies.columns:
        similar_movies=similar_movies.drop(movie,axis=1)
similar_movies.sum().sort_values(ascending=False).head()
