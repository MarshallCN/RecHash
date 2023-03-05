import random
from copy import deepcopy
from tqdm import tqdm
import os
import re

MAX_ITER = 10000
tem_ann_nodes = ['P', 'U']
user_nodes = ['U']
sem_nodes = ['U']
tem_nodes = ['V']
cw = 0


def determine_metapath(metapath, prs):
    if '(' in metapath and ')' in metapath:
        m = re.findall(r"\(([A-Za-z0-9_,]+)\)", metapath)
        for i, c in enumerate(m):
            pr = prs[i]
            if random.uniform(0, 1) < pr:
                metapath = metapath.replace('(%s)' % c, c[-1], 1)
            else:
                metapath = metapath.replace('(%s)' % c, c[0], 1)
        return metapath
    else:
        return metapath


def walk(relations, metapath, i, cur_node, s, fo):

    global tem_nodes, tem_ann_nodes, cw

    if i < len(metapath)-1:
        if True:
            if len(s) > 0:
                s = s + " " + str(cur_node)
            else:
                s = s + str(cur_node)

            candidate_nodes = dict()
            if cur_node in relations[metapath[i:i + 2]]:
                candidate_nodes[metapath[i:i + 2]] = list(deepcopy(relations[metapath[i:i + 2]][cur_node]))
            else:
                return

            if metapath[i] in tem_ann_nodes:
                for node_type in tem_nodes:
                    if cur_node in relations[metapath[i] + node_type].keys():
                        candidate_nodes[metapath[i] + node_type] = list(deepcopy(relations[metapath[i] + node_type][cur_node]))
                    else:
                        return

            cur_rela = metapath[i:i + 2]
            next_node = random.sample(candidate_nodes[cur_rela], 1).pop()

            # ----------------------- Adding temporal decay ----------------------------------
            """if cur_rela in ['PV', 'VP']:
                w = weights[cur_rela][cur_node]
                next_node = random.choices(candidate_nodes[cur_rela], weights=w, k=1).pop()
            else:
                next_node = random.sample(candidate_nodes[cur_rela], 1).pop()"""

            walk(relations, metapath, i+1, next_node, s, fo)

    else:
        s = s + " " + str(cur_node)
        s = s.split(" ")
        fo.write(s[0] + " " + s[-1] + '\n')
        cw += 1

def find_paths(dataset, relations, metapath, n_walk, prs):

    global MAX_ITER, cw

    m = metapath.replace('(', '').replace(',', '').replace(')', '')
    paths_file = "paths/%s/in_m%s_prs%s_nw%d.txt" % (dataset, m, "_".join([str(pr) for pr in prs]), n_walk)

    try:
        os.mkdir("paths/%s/"%dataset)
    except:
        pass

    if os.path.exists(paths_file):
        print('Paths file exists')
    else:
        fo = open(paths_file, 'w')
        cur_dict = relations['UP']
        for cur_node in tqdm(cur_dict.keys(), desc="Finding paths following meta-path %s" % metapath):
            cw = 0
            cl = 0
            while cw < n_walk and cl < MAX_ITER:
                s = ""
                metapath = determine_metapath(metapath, prs)
                walk(relations, metapath, 0, cur_node, s, fo)
                cl += 1

    print("Paths file: ", paths_file)
    return paths_file


def metapath2vec_embed(dataset, relations, metapath, n_walk, prs, emb_size):

    paths_file = find_paths(dataset, relations, metapath, n_walk, prs)
    embs_file = "embedding/%s/"%dataset + paths_file.split('/')[-1].replace('in', 'out').replace('.txt', '')

    try:
        os.mkdir("embedding/%s/"%dataset)
    except:
        pass

    if os.path.exists(embs_file):
        print('Embeddings file exists')
    else:
        print("Compute embeddings from: ", paths_file)
        print("Embedding size: ", emb_size)
        emb_size = str(emb_size)
        s ='./code_metapath2vec/metapath2vec -train %s -output %s -size %s' % (paths_file, embs_file, emb_size)
        embed_status = os.system(s)

    print("Embeddings file: %s.txt" % embs_file)
    return embs_file + ".txt"

