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


def get_items_same_block(p_j, blocks, all_items):
    items_same_block = set()
    for items in blocks.values():
        if p_j in items:
            items_same_block.update([_ for _ in items if _ in all_items])
    return list(items_same_block)


def get_train_matrix(train_set, all_users, all_items, file_emb, emb_method, p_j, blocks):
    train_matrix = list()
    # Consider only users in the same block
    selected_items = get_items_same_block(p_j, blocks, all_items)
    assert len(selected_items) > 0

    if emb_method == 0:
        train_matrix = np.zeros(shape=(len(selected_items), len(all_users)), dtype=np.dtype('u1'))
        for j, item in enumerate(selected_items):
            if item == p_j:
                target_ind = j
            for i, user in enumerate(all_users):
                if item in train_set[user]:
                    train_matrix[j, i] = 1
        return train_matrix, selected_items, target_ind

    elif emb_method in ["metapath2vec++", "deepwalk", "line", "node2vec", "sdne", "struc2vec"]:

        df_emb = get_embeddings(file_emb, emb_method).dropna()
        for j, item in enumerate(selected_items):
            if item == p_j:
                target_ind = j
            L = [userID for userID in all_users if item in train_set[userID]]
            user_embs = [df_emb[userID].values for userID in L if userID in set(df_emb.columns)]
            if len(user_embs) > 0:
                avg_user_emb = np.mean(user_embs, axis=0)
            else:
                avg_user_emb = np.zeros((df_emb.shape[0]))
            train_matrix.append(avg_user_emb)
        return np.array(train_matrix), selected_items, target_ind

def KNNItemBased(train_set, all_users, all_items, predict_users, n_item_neighbors, file_emb, emb_method, blocks_filepath):
    prediction = dict()

    with open(blocks_filepath, 'r') as fp:
        blocks = json.load(fp)

    for i, u_i in tqdm(predict_users, desc='predict recommendations'):

        score = dict()
        unseen_items = set(all_items) - set(train_set[u_i])

        for p_j in unseen_items:

            train_matrix, selected_items, target_ind = get_train_matrix(train_set, all_users, all_items, file_emb,
                                                                    emb_method, p_j, blocks)
            #print("Train matrix size: ", train_matrix.shape)
            model_knn = NearestNeighbors(metric='cosine', algorithm='auto')
            model_knn.fit(train_matrix)
            assert p_j == selected_items[target_ind]
            distances, indices = model_knn.kneighbors(train_matrix[target_ind].reshape(1, -1), n_neighbors=min(n_item_neighbors, train_matrix.shape[0]))
            similarity = (1 - distances[0])

            for k in range(1, len(indices[0])):
                p_k = all_items[indices[0][k]]
                if p_k in train_set[u_i]:
                    if p_j not in score.keys():
                        score[p_j] = 0
                    score[p_j] += similarity[k]

        sorted_pred = [k for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)]
        prediction[u_i] = sorted_pred[:100]

    return prediction




