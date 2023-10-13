import json
import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

base_src = '/Users/siyoung/Downloads'
rating_src = os.path.join(base_src, 'Rating.csv')
u_cols = ['user_id', 'work_id', 'rating']
ratings = pd.read_csv(rating_src,
                      sep=',',
                      names=u_cols,
                      encoding='latin-1')

ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')

ratings = ratings.set_index('user_id')
ratings = ratings.drop('user_id', axis=0)
ratings.head()

base_src = '/Users/siyoung/Downloads'
movie_src = os.path.join(base_src,'Movie.csv')
i_cols = ['work_id',
          'sf','action','adult','adventure','animation','comedy', 'criminal','documentary','drama','family ','fantasy',
          'horror','music','musical','mystery','performance','romance','thriller','variety','war','western',
        ]
movies = pd.read_csv(movie_src,
                    sep='|',
                    names=i_cols,
                    encoding='utf-8')
movies = movies.set_index('work_id')
movies = movies.drop('work_id',axis=0) # index 중복되어 삭제
movies.head() # 상위 5개 행 출력


######### neightbor size를 정해서 예측치를 계산하는 함수 #########
def CF_knn(user_id,work_id,neighbors_size=0):
  if int(work_id) in ratings_matrix.columns: # 해당 영화가 존재하면
    # 주어진 사용자(user_id)와 다른 사용자의 유사도 추출
    sim_scores = user_similarity[user_id].copy()
    # 주어진 영화(movie_id)와 다른 사용자의 유사도 추출
    movie_ratings = ratings_matrix[work_id].copy()
    # 주어진 영화에 대해서 평가하지 않은 사용자를 가중 평균 계산에서 제외하기 위해 인덱스 추출
    none_rating_idx = movie_ratings[movie_ratings.isnull()].index
    # 주어진 영화를 평가하지 않은 사용자와의 유사도 제거
    movie_ratings = movie_ratings.dropna()
    # 주어진 영화를 평가하지 않은 사용자와의 유사도 제거
    sim_scores = sim_scores.drop(none_rating_idx)

    ### neighbors_size가 지정되지 않은 경우 ###
    if neighbors_size == 0: # neighbor_size가 0이면 기존의 simple CF와 같음
      mean_rating = np.dot(sim_scores,movie_ratings) / sim_scores.sum()

    ### neighbors_size가 지정된 경우 ###
    else:
      if len(sim_scores) > 1:
        # neighbor_size와 sim_score 중에 작은 걸 택해야 분리 가능
        neighbors_size = min(neighbors_size,len(sim_scores))
        sim_scores = np.array(sim_scores)
        movie_ratings = np.array(movie_ratings)
        user_idx = np.argsort(sim_scores) #sim_scores 오름차순 인덱스
        sim_scores = sim_scores[user_idx][-neighbors_size:] # 정렬된 것을 뒤부터 뽑아냄
        movie_ratings = movie_ratings[user_idx][-neighbors_size:]
        mean_rating = np.dot(sim_scores,movie_ratings) / sim_scores.sum()
      else:
        mean_rating=3.0

  # train/test set의 분할에 따라 ratings_matrix에 해당 영화 없으면 기본값
  else:
    mean_rating=3.0
  return mean_rating


ratings_matrix = ratings.pivot_table(values='rating',
                              index='user_id',
                              columns='work_id')

### train set의 모든 가능한 사용자 pair의 코사인 유사도 계산 ###
matrix_dummy = ratings_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity,
                               index=ratings_matrix.index,
                               columns = ratings_matrix.index)

def recom_movie_by_CF_knn(user_id,n_items,neighbors_size=30):
  user_movie = ratings_matrix.loc[user_id].copy()

  for movie in ratings_matrix.columns:
    if pd.notnull(user_movie.loc[movie]): # 사용자가 해당 영화를 봤으면
      user_movie.loc[movie] = 0 # 추천 리스트에서 제외
    else:
      user_movie.loc[movie] = CF_knn(user_id,movie,neighbors_size)

  movie_sort = user_movie.sort_values(ascending=False)[:n_items] # 내림차순
  recom_movies = movies.loc[movie_sort.index] # 인덱스 반환
  return recom_movies


@app.route("/spring", methods=['POST'])
def spring():
    data_from_spring = request.data.decode('utf-8') # POST 요청에서 userId 추출
    userId = int(data_from_spring)
    print(userId)

    recommendations = recom_movie_by_CF_knn(user_id=str(userId), n_items= 10, neighbors_size=30)
    index_array = recommendations.index.to_numpy()
    recom_data = json.dumps(index_array.tolist())

    spring_api_url = "https://hs-ceos.shop/home/recommendation/works"
    response = requests.post(spring_api_url, data=recom_data)

    if response.status_code == 200:
        return jsonify({"result": "Data processed and sent to Spring successfully"})
    else:
        return jsonify({"error": "Failed to send data to Spring"}), 500


if __name__ == '__main__':
    app.run(debug=False,host="127.0.0.1",port=5000)