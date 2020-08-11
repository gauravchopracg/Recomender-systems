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


	

            
columns = pd.read_csv('columns.csv')
responses = pd.read_csv('responses.csv')

cols = ['History', 'Psychology', 'Politics', 'Mathematics', 'Physics', 'Internet', 'PC Software, Hardware', 'Economy, Management', 'Biology', 'Chemistry', 'Poetry reading', 'Geography', 'Foreign languages', 'Medicine', 'Law', 'Cars', 'Art', 'Religion', 'Outdoor activities', 'Dancing', 'Playing musical instruments', 'Poetry writing', 'Sport and leisure activities', 'Sport at competitive level', 'Gardening', 'Celebrity lifestyle', 'Shopping', 'Science and technology', 'Theatre', 'Socializing', 'Adrenaline sports', 'Pets']
needed1 = columns[columns.original.isin(cols)]
needed2 = needed1.short.tolist()
needed3 = responses[needed2]
needed3.dropna(inplace=True)
class2idxs = {i:k for i, k in enumerate(needed3)}

# preprocessing dataset to transform the variables into 
# dummify categorical features
career_metadata_selected_transformed = pd.get_dummies(needed3)
career_metadata_selected_transformed.head()

career_metadata_csr = csr_matrix(career_metadata_selected_transformed.values)
career_metadata_csr

model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)

model = model.fit(career_metadata_csr,
                  epochs=100,
                  num_threads=16, verbose=False)


product_train, product_test, product_users_altered = make_train(sparse_content_person, pct_test = 0.05)

model = implicit.als.AlternatingLeastSquares(factors=17,regularization=0.18, iterations=50)
alpha_val = 18
data_conf = (product_train * alpha_val).astype('double')
model.fit(data_conf)
person_vecs = model.user_factors
content_vecs = model.item_factors

print("----------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------")
print("printing auc_score")
print(calc_mean_auc(product_train, product_users_altered,[sparse.csr_matrix(person_vecs), sparse.csr_matrix(content_vecs.T)], product_test))
print("----------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------")
#msk = np.random.rand(len(grouped_df)) < 0.8
#train = df[msk]
#test = df[~msk]
#train, test = train_test_split(grouped_df, test_size=0.2)

#finding_similar_content
content_id = 450
n_similar = 10
similar = model.similar_items(content_id, n_similar)

print("----------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------")
print("printing similar content")
for content in similar:
    idx, score = content
    print(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])


#user_vecs, item_vecs = implicit_als(data_sparse, iterations=1, features=20, alpha_val=40)
print("----------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------")
#creating_recommendations_for users
person_vecs = sparse.csr_matrix(model.user_factors)
content_vecs = sparse.csr_matrix(model.item_factors)
person_id = 50
recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
print("----------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------")
print("printing user recommendations")
print(recommendations)
print("----------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------")