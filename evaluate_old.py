import json
import numpy as np
import recmetrics
from utility import swap_dict


def write_header(fp, dataset):
    if dataset == 'movielens':
        fp.write(
            "vtype\tgroup\t" +
            "avg_prec_1\tavg_prec_5\tavg_prec_10\tavg_prec_50\tavg_prec_100\t" +
            "avg_rec_1\tavg_rec_5\tavg_rec_10\tavg_rec_50\tavg_rec_100\t" +
            "avg_f1_1\tavg_f1_5\tavg_f1_10\tavg_f1_50\tavg_f1_100\t" +
            "norm_prec_1\tnorm_prec_5\tnorm_prec_10\tnorm_prec_50\tnorm_prec_100\t" +
            "norm_rec_1\tnorm_rec_5\tnorm_rec_10\tnorm_rec_50\tnorm_rec_100\t" +
            "norm_f1_1\tnorm_f1_5\tnorm_f1_10\tnorm_f1_50\tnorm_f1_100\t" +
            "item_cov_1\titem_cov_5\titem_cov_10\titem_cov_50\titem_cov_100\t" +
            "genre_cov_1\tgenre_cov_5\tgenre_cov_10\tgenre_cov_50\tgenre_cov_100\t" +
            "actor_cov_1\tactor_cov_5\tactor_cov_10\tactor_cov_50\tactor_cov_100\t" +
            "director_cov_1\tdirector_cov_5\tdirector_cov_10\tdirector_cov_50\tdirector_cov_100\t" +
            "tag_cov_1\ttag_cov_5\ttag_cov_10\ttag_cov_50\ttag_cov_100\t" +
            "novelty_1\tnovelty_5\tnovelty_10\tnovelty_50\tnovelty_100\t" +
            "avg_ep_1\tavg_ep_5\tavg_ep_10\tavg_ep_50\tavg_ep_100\t" +
            "avg_er_1\tavg_er_5\tavg_er_10\tavg_er_50\tavg_er_100\t" +
            "norm_ep_1\tnorm_ep_5\tnorm_ep_10\tnorm_ep_50\tnorm_ep_100\t" +
            "norm_er_1\tnorm_er_5\tnorm_er_10\tnorm_er_50\tnorm_er_100\t" +
            "\n"
        )
    elif dataset == 'amazon':
        fp.write(
            "vtype\tgroup\t" +
            "avg_prec_1\tavg_prec_5\tavg_prec_10\tavg_prec_50\tavg_prec_100\t" +
            "avg_rec_1\tavg_rec_5\tavg_rec_10\tavg_rec_50\tavg_rec_100\t" +
            "avg_f1_1\tavg_f1_5\tavg_f1_10\tavg_f1_50\tavg_f1_100\t" +
            "norm_prec_1\tnorm_prec_5\tnorm_prec_10\tnorm_prec_50\tnorm_prec_100\t" +
            "norm_rec_1\tnorm_rec_5\tnorm_rec_10\tnorm_rec_50\tnorm_rec_100\t" +
            "norm_f1_1\tnorm_f1_5\tnorm_f1_10\tnorm_f1_50\tnorm_f1_100\t" +
            "item_cov_1\titem_cov_5\titem_cov_10\titem_cov_50\titem_cov_100\t" +
            "category_cov_1\tcategory_cov_5\tcategory_cov_10\tcategory_cov_50\tcategory_cov_100\t" +
            "brand_cov_1\tbrand_cov_5\tbrand_cov_10\tbrand_cov_50\tbrand_cov_100\t" +
            "novelty_1\tnovelty_5\tnovelty_10\tnovelty_50\tnovelty_100\t" +
            "avg_ep_1\tavg_ep_5\tavg_ep_10\tavg_ep_50\tavg_ep_100\t" +
            "avg_er_1\tavg_er_5\tavg_er_10\tavg_er_50\tavg_er_100\t" +
            "norm_ep_1\tnorm_ep_5\tnorm_ep_10\tnorm_ep_50\tnorm_ep_100\t" +
            "norm_er_1\tnorm_er_5\tnorm_er_10\tnorm_er_50\tnorm_er_100\t" +
            "\n"
        )


def get_predicted_attributes(topKrec, relation):
    pred = []
    for row in topKrec:
        temp = []
        for itemID in row:
            if itemID in relation.keys():
                for v in relation[itemID]:
                    temp.append(v)
        pred.append(temp)
    return pred



def evaluate(dataset, prediction, relations, input_dir, test_data_ranges, listK, fres):

    with open(input_dir + "train.json", 'r') as fp:
        train_set = json.load(fp)

    item_key_train_set = swap_dict(train_set)
    pop = dict()
    for itemID, users in item_key_train_set.items():
        pop[itemID] = len(users)
    n_users = len(train_set)

    if dataset == 'movielens':
        genre_catalog = list(set(relations['GP'].keys()))
        actor_catalog = list(set(relations['AP'].keys()))
        director_catalog = list(set(relations['DP'].keys()))
        tag_catalog = list(set(relations['TP'].keys()))
    elif dataset == 'amazon':
        category_catalog = list(set(relations['CP'].keys()))
        brand_catalog = list(set(relations['BP'].keys()))
    item_catalog = list(set(relations['PU'].keys()))

    for r in test_data_ranges:
        if r == 'all':
            test_filename = input_dir + 'test.json'
        else:
            test_filename = input_dir + "test_%s.json" % (r)
        with open(test_filename, 'r') as fp:
            print("Test data: %s" % (test_filename))
            filtered_test_set = json.load(fp)
            filtered_test_users = [u for u in filtered_test_set.keys() if
                                   len(filtered_test_set[u]) >= 1 and u in train_set.keys()]
            assert set(filtered_test_users).issubset(set(train_set.keys()))
            print("#Test users: %d" % (len(filtered_test_users)))

        recommend = []
        actual = []
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

        avg_f1 = 2 * (np.array(avg_pre) * np.array(avg_rec)) / (np.array(avg_pre) + np.array(avg_rec))
        norm_f1 = 2 * (np.array(norm_pre) * np.array(norm_rec)) / (np.array(norm_pre) + np.array(norm_rec))

        write_header(fres, dataset)
        if dataset == 'movielens':
            fres.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                "%s" % r,
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
            fres.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                "%s" % r,
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

        print("MAP@K:\t\t%s" % "\t".join(np.char.mod('%f', np.array(avg_pre))))
        #print("MAR@K:\t\t%s" % "\t".join(np.char.mod('%f', np.array(avg_rec))))
        #print("Mean Precision@K:\t%s" % "\t".join(np.char.mod('%f', np.array(norm_pre))))
        print("Mean Recall@K:\t%s" % "\t".join(np.char.mod('%f', np.array(norm_rec))))
