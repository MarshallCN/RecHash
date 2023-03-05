import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def get_embeddings(filename):
    df_emb = pd.read_csv(filename, header=None, skiprows=1, delimiter=' ')
    df_emb = df_emb.T
    df_emb.columns = df_emb.iloc[0] # set the first row as column names
    return df_emb[1:-1].astype(float) # remove the first row


def get_users_same_block(u_i, blocks, all_users):
    users_same_block = set()
    for users in blocks.values():
        if u_i in users:
            users_same_block.update([_ for _ in users if _ in all_users])
    return list(users_same_block)


def get_train_matrix(train_set, all_users, all_items, file_emb, emb_method, u_i, hbs):
    train_matrix = list()
    # Consider only users in the same block
    selected_users = get_users_same_block(u_i, hbs, all_users)

    if emb_method == 0:
        train_matrix = np.zeros(shape=(len(selected_users), len(all_items)), dtype=np.dtype('u1'))
        for i, user in enumerate(selected_users):
            if user == u_i:
                target_ind = i
            for j, item in enumerate(all_items):
                if item in train_set[user]:
                    train_matrix[i, j] = 1
        return train_matrix, selected_users, target_ind

    elif emb_method == 1:
        df_emb = get_embeddings(file_emb).dropna()
        for i, user in enumerate(selected_users):
            if user == u_i:
                target_ind = i
            item_embs = [df_emb[itemID].values for itemID in train_set[user] if itemID in set(df_emb.columns)]
            if len(item_embs) > 0:
                avg_item_emb = np.mean(item_embs, axis=0)
            else:
                avg_item_emb = np.zeros((df_emb.shape[0]))
            train_matrix.append(avg_item_emb)
        return np.array(train_matrix), selected_users, target_ind


def KNNUserBased(train_set, all_users, all_items, predict_users, n_user_neighbors, file_emb, emb_method, blocks_filepath):
    prediction = dict()

    with open(blocks_filepath, 'r') as fp:
        blocks = json.load(fp)

    for i, u_i in tqdm(predict_users, desc='predict recommendations'):

        train_matrix, selected_users, target_ind = get_train_matrix(train_set, all_users, all_items, file_emb,
                                                                    emb_method, u_i, blocks)
        assert u_i == selected_users[target_ind] and len(selected_users) > 0
        #print("Train matrix size: ", train_matrix.shape)

        model_knn = NearestNeighbors(metric='cosine', algorithm='auto')
        model_knn.fit(train_matrix)
        distances, indices = model_knn.kneighbors(train_matrix[target_ind].reshape(1, -1),
                                                  n_neighbors=min(n_user_neighbors, train_matrix.shape[0]))
        similarity = (1 - distances[0])

        score = dict()
        unseen_items = set(all_items) - set(train_set[u_i])
        for j in range(1, len(indices[0])):
            u_j = selected_users[indices[0][j]]
            intersect_items = (set(train_set[u_j]).intersection(unseen_items))
            for item in intersect_items:
                if item not in score.keys():
                    score[item] = 0
                score[item] += similarity[j]
        sorted_pred = [k for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)]
        prediction[u_i] = sorted_pred[:100]

    return prediction

