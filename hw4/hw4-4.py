import pandas as pd

movies_df = pd.read_csv('movies.dat', sep='::')
ratings_df = pd.read_csv('ratings.dat',sep="::")

movie_ratings_df = movies_df.merge(ratings_df,how="inner",on=["movie_id,user_id"])