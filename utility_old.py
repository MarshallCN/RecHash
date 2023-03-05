import argparse

def swap_dict(old_dict):
    new_dict = {}
    for key, values in old_dict.items():
        for value in values:
            if value in new_dict.keys():
                new_dict[value].append(key)
            else:
                new_dict[value] = [key]
    return new_dict




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='',
                        help='Name of the dataset')
    parser.add_argument('--mode', nargs='?', default='user_based',
                        help='user_based: using user-based CF, item_based: using item-based CF')
    parser.add_argument('--emb_method', type=int, default=0,
                        help='0: using user-item interaction matrix, 1: using metapath2vec for generating user/item embeddings for CF-KNN')
    parser.add_argument('--blocks_metapath', nargs='?', default='',
                        help='User/Item blocks meta-path')


    # Parameters for KNN method
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='Number of user/item neighbors for CF')


    # paramters for metapath2vec (for ME approach only)
    parser.add_argument('--kg_file', nargs='?', default='relations_vbpr_100.json',
                        help='KG filename')
    parser.add_argument('--n_walk', type=int, default=100,
                        help='Maximum number of walks per starting node for metapath2vec (for ME only)')
    parser.add_argument('--embsize', type=int, default=300,
                        help='Node embedding size for metapath2vec (for ME only)')
    parser.add_argument('--metapath', nargs='?', default='',
                        help='Meta-path for generating node embeddings with metapath2vec (for ME only)')
    parser.add_argument('--prs', nargs='?', default='[0,0]',
                        help='Probabilities of going to visual node type when using visually-annotated meta-paths (for ME only)')

    # Evaluation
    parser.add_argument('--eval', type=bool, default=True,
                        help='True: perform evaluation, False: do not perform evaluation')
    parser.add_argument('--listK', nargs='?', default='[1, 5, 10, 50, 100]',
                        help='List of K for top-K recommendations')

    return parser.parse_args()