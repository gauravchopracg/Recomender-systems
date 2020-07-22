import random
import pandas as pd
import numpy as np
import implicit
from sklearn import metrics
import matplotlib.pylab as plt
import pickle
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import uvicorn
from fastapi import FastAPI

def make_train(ratings, pct_test = 0.2):
	test_set = ratings.copy() # Make a copy of the original set to be the test set.
	test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
	training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
	nonzero_inds = training_set.nonzero()
	nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
	random.seed(0)
	num_samples = int(np.ceil(pct_test*len(nonzero_pairs)))
	samples = random.sample(nonzero_pairs, num_samples)
	item_inds = [index[0] for index in samples] 
	user_inds = [index[1] for index in samples]
	training_set[item_inds, user_inds] = 0 
	training_set.eliminate_zeros()
	return training_set, test_set, list(set(user_inds))

def auc_score(predictions, test):
	fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
	return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_users, predictions, test_set):
	store_auc = []
	popularity_auc = []
	pop_items = np.array(test_set.sum(axis = 1)).reshape(-1)
	item_vecs = predictions[1]
	for user in altered_users:
		training_column = training_set[:,user].toarray().reshape(-1)
		#print(training_column)
		zero_inds = np.where(training_column == 0)
		user_vec = predictions[0][user,:]
		pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
		actual = test_set[:,user].toarray()[zero_inds,0].reshape(-1)
		pop = pop_items[zero_inds]
		store_auc.append(auc_score(pred, actual))
		pop = pop_items[zero_inds]
		store_auc.append(auc_score(pred, actual))
		popularity_auc.append(auc_score(pop, actual)) 
	return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  

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


	

            

articles_df = pd.read_csv('shared_articles.csv')
interactions_df = pd.read_csv('users_interactions.csv')
articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)
df = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')
#print(df['eventType'].value_counts())
#df.to_csv('D:\\1.csv')
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
}
df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])
df = df.drop_duplicates()
grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()
#print(grouped_df.head())
grouped_df['title'] = grouped_df['title'].astype("category")
grouped_df['personId'] = grouped_df['personId'].astype("category")
grouped_df['contentId'] = grouped_df['contentId'].astype("category")
grouped_df['person_id'] = grouped_df['personId'].cat.codes
grouped_df['content_id'] = grouped_df['contentId'].cat.codes
#grouped_df.to_csv('D:\\2.csv')
#train, validation, test = np.split(grouped_df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
sparse_person_content = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))
product_train, product_test, product_users_altered = make_train(sparse_content_person, pct_test = 0.05)

model = implicit.als.AlternatingLeastSquares(factors=17,regularization=0.18, iterations=45)
alpha_val = 18
data_conf = (product_train * alpha_val).astype('double')
model.fit(data_conf)
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
