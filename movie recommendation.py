# Content based Recommendation

import pandas as pd

movies_data = pd.read_csv("movies_metadata.csv")
print(movies_data.head())

# Explain in brief the purpose of TfidFVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = 'english')

movies_data["overview"] = movies_data["overview"].fillna('')

tfidf_matrix = tfidf.fit_transform(movies_data["overview"])

#print(tfidf_matrix.shape)


#print(tfidf.get_feature_names()[5000: 5010])

# Explain the notion of similarity between things here
from sklearn.metrics.pairwise import linear_kernel
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(movies_data.index, index = movies_data["title"]).drop_duplicates()

def get_recommendation(title, cosine_sim = cosine_similarity):
  idx = indices["title"]

  sim_scores = list(enumerate(cosine_sim[idx]))

  sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)

  sim_scores = sim_scores[1:11]

  movies_indices = [i[0] for i in sim_scores]

  return movies_data["title"].iloc[movies_indices]


print(get_recommendation("The Dark Knight Rises"))