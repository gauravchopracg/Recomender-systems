import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import heapq
from tqdm import tqdm
from random import shuffle


def load_dataset():

    
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
    df_train, df_test = train_test_split(grouped_df)

    # Create lists of all unique users and artists
    users = list(np.sort(grouped_df.person_id.unique()))
    items = list(np.sort(grouped_df.content_id.unique()))

    # Get the rows, columns and values for our matrix.
    rows = df_train.person_id.astype(int)
    cols = df_train.content_id.astype(int)

    values = list(df_train.eventStrength)

    # Get all user ids and item ids.
    uids = np.array(rows.tolist())
    iids = np.array(cols.tolist())

    # Sample 100 negative interactions for each user in our test data
    df_neg = get_negatives(uids, iids, items, df_test)

    return uids, iids, df_train, df_test, df_neg, users, items

def get_negatives(uids, iids, items, df_test):
   

    negativeList = []
    test_u = df_test['person_id'].values.tolist()
    test_i = df_test['content_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))
    zipped = set(zip(uids, iids))

    for (u, i) in test_ratings:
        negatives = []
        negatives.append((u, i))
        for t in range(100):
            j = np.random.randint(len(items)) # Get random item id.
            while (u, j) in zipped: # Check if there is an interaction
                j = np.random.randint(len(items)) # If yes, generate a new item id
            negatives.append(j) # Once a negative interaction is found we add it.
        negativeList.append(negatives)

    df_neg = pd.DataFrame(negativeList)

    return df_neg

def mask_first(x):
    """
    Return a list of 0 for the first item and 1 for all others
    """
    result = np.ones_like(x)
    result[0] = 0
    
    return result
   
def train_test_split(df):
    

    # Create two copies of our dataframe that we can modify
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # Group by user_id and select only the first item for
    # each user (our holdout).
    df_test = df_test.groupby(['person_id']).first()
    df_test['person_id'] = df_test.index
    df_test = df_test[['person_id', 'content_id', 'eventStrength']]
    del df_test.index.name

    # Remove the same items as we for our test set in our training set.
    mask = df.groupby(['person_id'])['person_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test

def get_train_instances():
     

     user_input, item_input, labels = [],[],[]
     zipped = set(zip(uids, iids))

     for (u, i) in zip(uids,iids):
         # Add our positive interaction
         user_input.append(u)
         item_input.append(i)
         labels.append(1)

         # Sample a number of random negative interactions
         for t in range(num_neg):
             j = np.random.randint(len(items))
             while (u, j) in zipped:
                 j = np.random.randint(len(items))
             user_input.append(u)
             item_input.append(j)
             labels.append(0)

     return user_input, item_input, labels


def random_mini_batches(U, I, L, mini_batch_size=256):
    

    mini_batches = []

    shuffled_U, shuffled_I, shuffled_L = shuffle(U, I, L)

    num_complete_batches = int(math.floor(len(U)/mini_batch_size))
    for k in range(0, num_complete_batches):
        mini_batch_U = shuffled_U[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_I = shuffled_I[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
        mini_batches.append(mini_batch)

    if len(U) % mini_batch_size != 0:
        mini_batch_U = shuffled_U[num_complete_batches * mini_batch_size: len(U)]
        mini_batch_I = shuffled_I[num_complete_batches * mini_batch_size: len(U)]
        mini_batch_L = shuffled_L[num_complete_batches * mini_batch_size: len(U)]

        mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
        mini_batches.append(mini_batch)

    return mini_batches


def get_hits(k_ranked, holdout):
    

    for item in k_ranked:
        if item == holdout:
            return 1
    return 0


def eval_rating(idx, test_ratings, test_negatives, K):
    

    map_item_score = {}

    # Get the negative interactions our user.
    items = test_negatives[idx]

    # Get the user idx.
    user_idx = test_ratings[idx][0]

    # Get the item idx, i.e. our holdout item.
    holdout = test_ratings[idx][1]

    # Add the holdout to the end of the negative interactions list.
    items.append(holdout)

    # Prepare our user and item arrays for tensorflow.
    predict_user = np.full(len(items), user_idx, dtype='int32').reshape(-1,1)
    np_items = np.array(items).reshape(-1,1)

    # Feed user and items into the TF graph .
    predictions = session.run([output_layer], feed_dict={user: predict_user, item: np_items})

    # Get the predicted scores as a list
    predictions = predictions[0].flatten().tolist()

    # Map predicted score to item id.
    for i in range(len(items)):
        current_item = items[i]
        map_item_score[current_item] = predictions[i]

    # Get the K highest ranked items as a list
    k_ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    # Get a list of hit or no hit.   
    hits = get_hits(k_ranked, holdout)

    return hits


def evaluate(df_neg, K=10):
    

    hits = []

    test_u = df_test['person_id'].values.tolist()
    test_i = df_test['content_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))

    df_neg = df_neg.drop(df_neg.columns[0], axis=1)
    test_negatives = df_neg.values.tolist()

    for idx in range(len(test_ratings)):
        
        hitrate = eval_rating(idx, test_ratings, test_negatives, K)
        hits.append(hitrate)

    return hits

# Load and prepare our data.
uids, iids, df_train, df_test, df_neg, users, items = load_dataset()


#-------------
# HYPERPARAMS
#-------------

num_neg = 6
latent_features = 20
epochs = 400
batch_size = 256
learning_rate = 0.015


#-------------------------
# TENSORFLOW GRAPH
#-------------------------

graph = tf.Graph()

with graph.as_default():

    # Define input placeholders for user, item and label.
    user = tf.placeholder(tf.int32, shape=(None, 1))
    item = tf.placeholder(tf.int32, shape=(None, 1))
    label = tf.placeholder(tf.int32, shape=(None, 1))

    # User embedding for MLP
    mlp_u_var = tf.Variable(tf.random_normal([len(users), 32], stddev=0.05),
            name='mlp_user_embedding')
    mlp_user_embedding = tf.nn.embedding_lookup(mlp_u_var, user)

    # Item embedding for MLP
    mlp_i_var = tf.Variable(tf.random_normal([len(items), 32], stddev=0.05),
            name='mlp_item_embedding')
    mlp_item_embedding = tf.nn.embedding_lookup(mlp_i_var, item)

    # User embedding for GMF
    gmf_u_var = tf.Variable(tf.random_normal([len(users), latent_features],
        stddev=0.05), name='gmf_user_embedding')
    gmf_user_embedding = tf.nn.embedding_lookup(gmf_u_var, user)

    # Item embedding for GMF
    gmf_i_var = tf.Variable(tf.random_normal([len(items), latent_features],
        stddev=0.05), name='gmf_item_embedding')
    gmf_item_embedding = tf.nn.embedding_lookup(gmf_i_var, item)

    # Our GMF layers
    gmf_user_embed = tf.keras.layers.Flatten()(gmf_user_embedding)
    gmf_item_embed = tf.keras.layers.Flatten()(gmf_item_embedding)
    gmf_matrix = tf.multiply(gmf_user_embed, gmf_item_embed)

    # Our MLP layers
    mlp_user_embed = tf.keras.layers.Flatten()(mlp_user_embedding)
    mlp_item_embed = tf.keras.layers.Flatten()(mlp_item_embedding)
    mlp_concat = tf.keras.layers.concatenate([mlp_user_embed, mlp_item_embed])

    mlp_dropout = tf.keras.layers.Dropout(0.2)(mlp_concat)

    mlp_layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')(mlp_dropout)
    mlp_batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_1)
    mlp_dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm1)

    mlp_layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')(mlp_dropout1)
    mlp_batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_2)
    mlp_dropout2 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm2)

    mlp_layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')(mlp_dropout2)
    mlp_layer_4 = tf.keras.layers.Dense(8, activation='relu', name='layer4')(mlp_layer_3)

    
    merged_vector = tf.keras.layers.concatenate([gmf_matrix, mlp_layer_4])

     
    output_layer = tf.keras.layers.Dense(1,
            kernel_initializer="lecun_uniform",
            name='output_layer')(merged_vector)

     
    loss = tf.losses.sigmoid_cross_entropy(label, output_layer)

    
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
    step = opt.minimize(loss)

    
    init = tf.global_variables_initializer()


session = tf.Session(config=None, graph=graph)
session.run(init)

for epoch in range(epochs):

    
    user_input, item_input, labels = get_train_instances()

    
    feed_dict = {user: np.array(user_input).reshape(-1,1),
                    item: np.array(item_input).reshape(-1,1),
                    label: np.array(labels).reshape(-1,1)}
   
       
    _, l = session.run([step, loss], feed_dict)

        
    #progress.update(1)
    #progress.set_description('Epoch: %d - Loss: %.3f' % (epoch+1, l))

    #progress.close()

hits = evaluate(df_neg)
print(np.array(hits).mean())
