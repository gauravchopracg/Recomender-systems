import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import pandas as pd
import numpy as np
import implicit
from sklearn import metrics
import matplotlib.pylab as plt

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
articles_df = pd.read_csv('shared_articles.csv')
interactions_df = pd.read_csv('users_interactions.csv')
articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)
df = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')
event_type_strength = {'VIEW': 1.0,'LIKE': 2.0, 'BOOKMARK': 3.0, 'FOLLOW': 4.0,'COMMENT CREATED': 5.0,  }
df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])
df = df.drop_duplicates()
grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()
grouped_df['title'] = grouped_df['title'].astype("category")
grouped_df['personId'] = grouped_df['personId'].astype("category")
grouped_df['contentId'] = grouped_df['contentId'].astype("category")
grouped_df['person_id'] = grouped_df['personId'].cat.codes
grouped_df['content_id'] = grouped_df['contentId'].cat.codes
sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
sparse_person_content = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))
person_vecs = sparse.csr_matrix(model.user_factors)
content_vecs = sparse.csr_matrix(model.item_factors)

def recommend(person_id, sparse_person_content, person_vecs, content_vecs, num_contents=10):
    person_interactions = sparse_person_content[person_id,:].toarray()
    person_interactions = person_interactions.reshape(-1) + 1
    person_interactions[person_interactions > 1] = 0
    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = person_interactions * rec_vector_scaled
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]
    
    
    titles = []
    scores = []

    for idx in content_idx:
        
        titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    return recommendations

@app.route('/')
def predict():
    
    
    person_vecs = sparse.csr_matrix(model.user_factors)
    content_vecs = sparse.csr_matrix(model.item_factors)
    person_id = 50
    recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
    return render_template('index.html', prediction_text= recommendations)



if __name__ == "__main__":
    app.run(debug=True)
