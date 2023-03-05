import json
import numpy as np
import recmetrics
from utility import get_predicted_attributes, swap_dict


listK = [1, 5, 10, 50, 100]

def evaluate(prediction, dataset, input_dir, all_users, all_items, fres, vtype, rank_weights_prec=True, rank_weights_rec=False):
    if dataset == 'movielens':
        ranges = ['2to10', '10to20', '20to50', '50to100', '100to1276', 'all']
    elif dataset =='amazon':
        ranges = ['2to10', '10to20', '20to50', '50to100', 'all']

    with open(input_dir + "train.json", 'r') as fp:
        train_set = json.load(fp)
    with open(input_dir + 'kg/relations_vbpr_100.json', 'r') as fp:
        relations = json.load(fp)

    item_key_train_set = swap_dict(train_set)
    pop = dict()
    for itemID, users in item_key_train_set.items():
        pop[itemID] = len(users)
    n_users = len(train_set)

    user_remap = dict(zip(all_users, range(len(all_users))))
    item_remap = dict(zip(all_items, range(len(all_items))))

    if dataset == 'movielens':
        genre_catalog = list(set(relations['GP'].keys()))
        actor_catalog = list(set(relations['AP'].keys()))
        director_catalog = list(set(relations['DP'].keys()))
        tag_catalog = list(set(relations['TP'].keys()))
    elif dataset == 'amazon':
        category_catalog = list(set(relations['CP'].keys()))
        brand_catalog = list(set(relations['BP'].keys()))
    item_catalog = list(set(relations['PU'].keys()))

    for r in ranges:
        if r == 'all':
            test_filename = input_dir + 'test.json'
        else:
            test_filename = input_dir + "test_%s.json" % (r)
        with open(test_filename, 'r') as fp:
            #print("Reading test data from \t\t%s" % (test_filename))
            filtered_test_set = json.load(fp)
            filtered_test_users = [u for u in filtered_test_set.keys() if
                                   len(filtered_test_set[u]) >= 1 and u in all_users]
            assert set(filtered_test_users).issubset(set(all_users))

        #print("#Training users: %d\n#Filtered test users: %d" % (len(train_set), len(filtered_test_users)))

        recommend = []
        actual = []
        expl = []
        bought_expl = []
        for userID in filtered_test_users:
          if userID in prediction:
            recommend.append(prediction[userID])
            assert len(filtered_test_set[userID]) > 0
            actual.append(filtered_test_set[userID])


        avg_rec = []
        avg_pre = []
        norm_rec = []
        norm_pre = []

        item_cov = []
        genre_cov = []
        actor_cov = []
        director_cov = []
        tag_cov = []
        category_cov = []
        brand_cov = []
        nov = []
        avg_ep, avg_er, norm_er, norm_ep = [], [], [], []
        normal_ep = []
        for K in listK:

            topKrec = [temp[:K] for temp in recommend]

            avg_pre.extend([recmetrics.mapk(actual, topKrec, k=K, rank_weights=True)])
            avg_rec.extend([recmetrics.mark(actual, topKrec, k=K, rank_weights=True)])
            norm_pre.extend([recmetrics.mapk(actual, topKrec, k=K, rank_weights=False)])
            norm_rec.extend([recmetrics.mark(actual, topKrec, k=K, rank_weights=False)])

            item_cov.extend([recmetrics.prediction_coverage(topKrec, item_catalog)])

            if dataset == 'movielens':
                genre_pred = get_predicted_attributes(topKrec, relations['PG'])
                genre_cov.extend([recmetrics.prediction_coverage(genre_pred, genre_catalog)])
                actor_pred = get_predicted_attributes(topKrec, relations['PA'])
                actor_cov.extend([recmetrics.prediction_coverage(actor_pred, actor_catalog)])
                director_pred = get_predicted_attributes(topKrec, relations['PD'])
                director_cov.extend([recmetrics.prediction_coverage(director_pred, director_catalog)])
                tag_pred = get_predicted_attributes(topKrec, relations['PT'])
                tag_cov.extend([recmetrics.prediction_coverage(tag_pred, tag_catalog)])
            elif dataset == 'amazon':
                category_pred = get_predicted_attributes(topKrec, relations['PC'])
                category_cov.extend([recmetrics.prediction_coverage(category_pred, category_catalog)])
                brand_pred = get_predicted_attributes(topKrec, relations['PB'])
                brand_cov.extend([recmetrics.prediction_coverage(brand_pred, brand_catalog)])

            nov.extend([recmetrics.novelty(topKrec, pop, n_users, K)[0]])



        #print("avg. precision:\t%s" % "\t".join(np.char.mod('%f', np.array(avg_pre))))
        #print("avg. recall:\t%s" % "\t".join(np.char.mod('%f', np.array(avg_rec))))

        avg_f1 = 2 * (np.array(avg_pre) * np.array(avg_rec)) / (np.array(avg_pre) + np.array(avg_rec))
        norm_f1 = 2 * (np.array(norm_pre) * np.array(norm_rec)) / (np.array(norm_pre) + np.array(norm_rec))

        if dataset == 'movielens':
            fres.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                vtype, "%s" % r,
                "\t".join(np.char.mod('%f', np.array(avg_pre))),
                "\t".join(np.char.mod('%f', np.array(avg_rec))),
                "\t".join(np.char.mod('%f', np.array(avg_f1))),
                "\t".join(np.char.mod('%f', np.array(norm_pre))),
                "\t".join(np.char.mod('%f', np.array(norm_rec))),
                "\t".join(np.char.mod('%f', np.array(norm_f1))),
                "\t".join(np.char.mod('%.2f', np.array(item_cov))),
                "\t".join(np.char.mod('%.2f', np.array(genre_cov))),
                "\t".join(np.char.mod('%.2f', np.array(actor_cov))),
                "\t".join(np.char.mod('%.2f', np.array(director_cov))),
                "\t".join(np.char.mod('%.2f', np.array(tag_cov))),
                "\t".join(np.char.mod('%.2f', np.array(nov)))
            ))
        elif dataset == 'amazon':
            fres.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                vtype, "%s" % r,
                "\t".join(np.char.mod('%f', np.array(avg_pre))),
                "\t".join(np.char.mod('%f', np.array(avg_rec))),
                "\t".join(np.char.mod('%f', np.array(avg_f1))),
                "\t".join(np.char.mod('%f', np.array(norm_pre))),
                "\t".join(np.char.mod('%f', np.array(norm_rec))),
                "\t".join(np.char.mod('%f', np.array(norm_f1))),
                "\t".join(np.char.mod('%.2f', np.array(item_cov))),
                "\t".join(np.char.mod('%.2f', np.array(category_cov))),
                "\t".join(np.char.mod('%.2f', np.array(brand_cov))),
                "\t".join(np.char.mod('%.2f', np.array(nov)))
            ))
