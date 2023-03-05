import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def get_embeddings(filename, method):

    if method in ["metapath2vec++", "metapath2vec++ no context", "TNE", "TNE no context"]:
        df_emb = pd.read_csv(filename, header=None, skiprows=1, delimiter=' ')
        df_emb = df_emb.T
        df_emb.columns = df_emb.iloc[0] # set the first row as column names
        return df_emb[1:-1].astype(float) # remove the first row

    elif method in ["deepwalk", "line", "node2vec", "sdne", "struc2vec"]:
        return pd.read_csv(filename, delimiter=',').astype(float)


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


def swap_dict(old_dict):
    new_dict = {}
    for key, values in old_dict.items():
        for value in values:
            if value in new_dict.keys():
                new_dict[value].append(key)
            else:
                new_dict[value] = [key]
    return new_dict


def see_dist(data, plot_name):
    avg = sum([len(values) for values in data.values()]) / len(data)
    maxi = max([len(values) for values in data.values()])
    mini = min([len(values) for values in data.values()])
    plt.figure()
    sns.distplot(np.array([len(values) for values in data.values()]), kde=False, axlabel='N')
    plt.title("%s\nMin = %d, Max = %d, Avg = %.2f" % (plot_name, mini, maxi, avg))
    plt.savefig(plot_name.replace(' ', '_'))
    plt.close()


def kg_analyse(relations, dataset, vtype=''):

    if dataset == 'amazon':
        with open("kg/%s/kg_summary_%s.txt" % (dataset, vtype), 'w') as fp:
            fp.write("#U nodes\t%d\n" % len(relations['UP']))
            fp.write("#P nodes\t%d\n" % len(relations['PU']))
            fp.write("#C nodes\t%d\n" % len(relations['CP']))
            fp.write("#B nodes\t%d\n" % len(relations['BP']))
            fp.write("#T nodes\t%d\n" % len(relations['TP']))
            #fp.write("#A nodes\t%d\n" % len(relations['AP']))
            #fp.write("#W nodes\t%d\n" % len(relations['WP']))
            #fp.write("# (%s) nodes\t%d\n" % (vtype, len(relations['VP'])))

            total_edges = 0
            total_edges_no_visual = 0
            for r in relations.keys():
                count_edges = sum([len(nodes) for nodes in relations[r].values()])
                fp.write("#%s-%s edges\t%d\n" % (r[0], r[1], count_edges))
                total_edges += count_edges
                if r[0] != 'V' and r[1]!= 'V':
                    total_edges_no_visual += count_edges

            for r in relations.keys():
                try:
                    fp.write("Avg. %s-%s\t%.2f\n" % (r[0], r[1], sum([len(nodes) for nodes in relations[r].values()]) / len(relations[r])))
                except:
                    pass

            total_nodes = len(relations['UP']) + len(relations['PU']) + len(relations['CP']) + len(relations['BP']) + len(relations['TP'])
            total_edges = total_edges
            total_edges_no_visual = total_edges_no_visual
            fp.write("Total nodes: %d\n" % total_nodes)
            fp.write("Total nodes: %d\n" % total_edges)
            fp.write("Total nodes: %d\n" % total_edges_no_visual)

    elif dataset == 'movielens':
        with open("kg/%s/kg_summary_%s.txt" % (dataset, vtype), 'w') as fp:
            fp.write("#U nodes\t%d\n" % len(relations['UP']))
            fp.write("#P nodes\t%d\n" % len(relations['PU']))
            fp.write("#G nodes\t%d\n" % len(relations['GP']))
            fp.write("#A nodes\t%d\n" % len(relations['AP']))
            fp.write("#D nodes\t%d\n" % len(relations['DP']))
            fp.write("#T nodes\t%d\n" % len(relations['TP']))
            #fp.write("#V (%s) nodes\t%d\n" % (vtype, len(relations['VP'])))

            for r in relations.keys():
                fp.write("#%s-%s edges\t%d\n" % (r[0], r[1], sum([len(nodes) for nodes in relations[r].values()])))

            for r in relations.keys():
                fp.write("Avg. %s-%s\t%.2f\n" % (r[0], r[1], sum([len(nodes) for nodes in relations[r].values()]) / len(relations[r])))

        try:
            os.mkdir("kg/%s/%s" % (dataset, vtype))
        except:
            pass

        for r in relations.keys():
            see_dist(relations[r], "kg/%s/%s/dist%s_%s.jpg" % (dataset, vtype, r, vtype) )


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