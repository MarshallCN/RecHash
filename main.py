import random
import os
import json
import tracemalloc
import time
from UserKNN import KNNUserBased
from itemKNN import KNNItemBased
from node_embed import metapath2vec_embed
from utility_old import parse_args
from evaluate import evaluate

if __name__ == '__main__':

    ############################## Setting up parameters ###################################
    random.seed(10)
    args = parse_args()
    if args.dataset == 'movielens':
        input_dir = "dataset/movielens/2core/"
        result_dir = 'predictions/movielens/'
        blocks_dir = 'dataset/movielens/2core/blocks/'
        test_data_ranges = ['2to10', '10to20', '20to50', '50to100', '100to1276', 'all']
        short_name = 'ml'
    elif args.dataset == 'amazon':
        input_dir = "dataset/amazon/2core_clothing/"
        result_dir = 'predictions/amazon/'
        blocks_dir = 'dataset/amazon/2core_clothing/blocks/'
        test_data_ranges = ['2to10', '10to20', '20to50', '50to100', '100to137', 'all']
        short_name = 'az'
    else:
        raise NotImplementedError

    blocks_filepath = blocks_dir + '%s.json' % args.blocks_metapath

    try:
        os.mkdir(result_dir)
    except:
        pass


    ###################### Reading input data and KG relations ###########################
    print("#### Reading input data and KG relations")
    with open(input_dir + "train.json", 'r') as fp:
        print("Reading training data from \t%s" % (input_dir + "train.json"))
        train_set = json.load(fp)
    with open(input_dir + "test.json", 'r') as fp:
        print("Reading test data from \t\t%s" % (input_dir + "test.json"))
        test_set = json.load(fp)
    with open(input_dir + 'kg/%s' % args.kg_file, 'r') as fp:
        relations = json.load(fp)

    def sample_from_dict(d, sample=10):
        random.seed(0)
        keys = random.sample(list(d), sample)
        values = [d[k] for k in keys]
        return dict(zip(keys, values))
    if args.dataset == 'amazon':
      test_set = sample_from_dict(test_set,576)
    
    all_users = list(train_set.keys())
    train_items = set([itemID for items in train_set.values() for itemID in items])
    test_items = set([itemID for items in test_set.values() for itemID in items])
    all_items = list(train_items.union(test_items))
    predict_users = [(ind, u) for ind, u in enumerate(all_users) if u in test_set.keys() and len(test_set[u]) >= 1]
    print("#Users %d\t#items %d" % (len(train_set), len(all_items)))
    print("#Training users: %d\n#Test users: %d" % (len(train_set), len(predict_users)))
    print('')


    ####################### Generating user/item embeddings (representations) ################################
    print("##### Generating user/item embeddings")
    tracemalloc.start()
    start = time.time()
    print("Embedding method (0: User-Item interaction matrix, 1: metapath2vec):  ", args.emb_method)

    if args.emb_method == 0:
        # Use user-item interation matrix for user/item embeddings
        file_emb = '-'
    elif args.emb_method == 1:
        # Use metapath2vec_embed to generate node embeddings and write the embeddings to file_emb.txt
        file_emb = metapath2vec_embed(args.dataset, relations, args.metapath, args.n_walk, eval(args.prs), args.embsize)
        print("Embedding file: ", file_emb.split('/')[-1])
    else:
        raise NotImplementedError
    print('')

    ################################## Predicting ######################################
    print("##### Predicting")
    print(args.mode)

    print("CF-KNN method: ", args.mode)
    if args.emb_method == 0:
        # Use user-item interation matrix for user/item embeddings (representations)
        result_filename = "RH_%s_%s_bm%s_nb%d" % (args.mode, short_name, args.blocks_metapath, args.n_neighbors)
    elif args.emb_method == 1:
        result_filename = "ME_%s_%s_bm%s_nb%d_m%s_nw%d_emb%d" % (args.mode, short_name, args.blocks_metapath, args.n_neighbors, args.metapath, args.n_walk, args.embsize)

    if args.mode == 'user_based':
        prediction = KNNUserBased(train_set, all_users, all_items, predict_users, args.n_neighbors, file_emb, args.emb_method, blocks_filepath)
    elif args.mode == 'item_based':
        prediction = KNNItemBased(train_set, all_users, all_items, predict_users, args.n_neighbors, file_emb, args.emb_method, blocks_filepath)
    else:
        raise NotImplementedError

    with open('predictions/%s/%s.json' % (args.dataset, result_filename), 'w') as fp:
        json.dump(prediction, fp)

    end = time.time()
    print('')

    ################################## Evaluating ######################################
    # print("##### Evaluating")

    # # Accuracy, Novelty, Diversity
    # if args.eval == True:
    #     with open("predictions/%s/%s_eval.csv"%(args.dataset, result_filename), 'a') as fres:
    #         evaluate(args.dataset, prediction, relations, input_dir, test_data_ranges, eval(args.listK), fres)

    # # Efficiency, Scalability
    # print("Run Time is:", end - start)
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    # tracemalloc.stop()

    # with open("predictions/%s/%s_perf.csv" % (args.dataset, result_filename), 'a') as fp:
    #     fp.write("Label,RunTime,CMemory,PMemory\n")
    #     fp.write("%s,%f,%f,%f\n" % (
    #     result_filename.replace('.csv', ''), (end - start), (current / 10 ** 6), (peak / 10 ** 6)))