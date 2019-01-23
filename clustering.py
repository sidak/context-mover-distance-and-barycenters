from __future__ import division
from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np
import sys, codecs, json
import pickle
import os
import argparse
import logging
import time
import utils
from cooc_analysis import Analysis
import random

'''
Usage

python clustering.py 
    --input-vector-file ./data/vectors/entailment_vectors_200.glove 
    --algo kmeans 
    --clusters-to-make 100 
    --n-words 80000 
    --target-folder ./data/vectors/clusters/entailment_vectors_200
    --dim 200
'''

'''
Usage (in Vocab mode, with pretrained vectors)

python clustering.py 
    --input-vector-file ./data/vectors/entailment_vectors_200.glove
    --algo kmeans 
    --clusters-to-make 100 
    --vocab-file ./data/vectors/entailment_vocab_200.glove
    --target-folder ./data/vectors/clusters/entailment_vectors_200
    --dim 200
'''


def build_word_vector_matrix(vector_file, n_words, dim=300, word2id=None, context_vectors=False, no_lower=False):
    '''Iterate over the GloVe array read from sys.argv[1] and return its vectors and labels as arrays'''
    np_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'latin1') as f:
        for c, r in enumerate(f):
            if no_lower:
                sr = r.split()
            else:
                sr = r.lower().split()

            if (not context_vectors) and len(sr) != (dim + 1):
                # dim + 1 is because it first reads the word (label and then dim many numbers)
                # just removes the vector for white space 
                # (it actually reads it twice, once for space and then vector)
                continue

            if context_vectors and len(sr) != (2 * dim + 3):
                print("Context vector: Not in appropriate format ")
                continue

            try:
                st_idx = 1
                if context_vectors:
                    st_idx += dim + 1  # exclude the target vector and its bias

                if word2id is not None:
                    if sr[0] in word2id:
                        labels_array.append(sr[0])
                        np_arrays.append(np.array([float(i) for i in sr[st_idx:st_idx + dim]]))
                else:
                    labels_array.append(sr[0])
                    np_arrays.append(np.array([float(i) for i in sr[st_idx:st_idx + dim]]))

                # since all words from the given vocabulary many not possess a word vector
                # as a result maintain a vocab_covered dictionary

            except ValueError:
                print(c, len(sr))

            if c == (n_words + 1):
                return np.array(np_arrays), labels_array

    return np.array(np_arrays), labels_array


def handle_empty_clusters(cluster_to_words, num_clusters):
    new_cluster_to_words = {}
    empty_cluster_ids = []
    for i in range(num_clusters):
        cl_key = str(i)
        if cl_key in cluster_to_words:
            new_cluster_to_words[cl_key] = cluster_to_words[cl_key]
        else:
            print("Encountered empty cluster with id: ", str(i))
            new_cluster_to_words[cl_key] = []
            empty_cluster_ids.append(i)
    return new_cluster_to_words, empty_cluster_ids


def handle_empty_clusters_better(cluster_to_words, num_clusters):
    sorted_cluster_to_words = sorted(list(cluster_to_words.items()),
                                     key=lambda item: int(item[0]))  # sorted list of tuples
    sorted_word_members = list(zip(*sorted_cluster_to_words))[
        1]  # 'zip(*)' unzips & we have a list containing 2 tuples, list[0] = idxs, list[0]=vals
    new_cluster_ids = [str(clus_id) for clus_id in range(len(sorted_word_members))]
    new_cluster_to_words = dict(zip(new_cluster_ids, sorted_word_members))
    print("Sorted cluster_to_words now, rechecking len of it ", len(sorted_word_members))

    empty_cluster_ids = []
    for i in range(num_clusters):
        cl_key = str(i)
        if cl_key not in cluster_to_words:
            print("Encountered empty cluster with id: ", str(i))
            empty_cluster_ids.append(i)

    return new_cluster_to_words, empty_cluster_ids


def find_word_clusters(labels_array, cluster_labels):
    '''Read in the labels array and clusters label and return the set of words in each cluster'''
    cluster_to_words = {}
    for idx, word in enumerate(cluster_labels):
        if (not str(word) in cluster_to_words):
            cluster_to_words[str(word)] = []
        cluster_to_words[str(word)].append(labels_array[idx])

    return cluster_to_words


def visualize_clustering(cluster_to_words, word_vector_dict, cluster_center_vectors, points_per_cluster=5,
                         perplexity=30, n_components=2, init='pca', \
                         n_iter=2500, random_state=23, global_step=-1, tensorboard=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import io
    from PIL import Image
    print("IN VISUALIZATION")

    clusters_to_make = len(cluster_to_words)
    closest_pts_dic = {}

    for i in range(clusters_to_make):
        cluster_key = str(i)
        points = cluster_to_words[cluster_key]
        vectors = np.array([word_vector_dict[point] for point in points], dtype=np.float32)
        cluster_center = cluster_center_vectors[int(cluster_key)]
        vectors = vectors - cluster_center
        distances = np.linalg.norm(vectors, axis=1)
        indices = np.argsort(distances)
        closest_pts = [points[idx] for idx in indices]
        closest_selected_pts = closest_pts[:points_per_cluster]
        closest_pts_dic[cluster_key] = closest_selected_pts

    vis_word_labels = []
    vis_vectors = []
    vis_words = []
    for i in range(clusters_to_make):
        closest_len = len(closest_pts_dic[str(i)])
        vis_word_labels += [i] * min(points_per_cluster, closest_len)
        vis_vectors += [word_vector_dict[point] for point in closest_pts_dic[str(i)]]
        vis_words += closest_pts_dic[str(i)]

    vis_vectors = np.asarray(vis_vectors)

    ha_vals = ['center', 'right', 'left']
    va_vals = ['center', 'top', 'bottom']

    ha_va_vals = [(x, y) for x in ha_vals for y in va_vals]
    ha_va_vals.remove((ha_vals[0], va_vals[0]))
    random.shuffle(ha_va_vals)

    tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter,
                      random_state=random_state)
    low_dim_vectors = tsne_model.fit_transform(vis_vectors)
    x_comp = low_dim_vectors[:, 0]
    y_comp = low_dim_vectors[:, 1]

    fig = plt.figure(figsize=(12, 12))
    plt.scatter(x_comp, y_comp, c=vis_word_labels)

    global_ct = 0
    for i in range(clusters_to_make):
        closest_len = len(closest_pts_dic[str(i)])
        local_vis_words = closest_pts_dic[str(i)]

        for j in range(closest_len):
            plt.annotate(local_vis_words[j],
                         xy=(x_comp[global_ct + j], y_comp[global_ct + j]),
                         xytext=(10, 10),
                         textcoords='offset points',
                         ha=ha_va_vals[j][0],
                         va=ha_va_vals[j][1])

        global_ct += closest_len

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if tensorboard is not None and global_step != -1:
        print("Adding image to tensorboard")
        tensorboard.add_image('images/tSNE_Clustering', image, global_step=global_step)
    else:
        print("Tensorboard is not available for saving cluster visualization")


def perform_clustering(data_vec, words, num_clusters, lib='sklearn', cores='24', metric='euclidean', tolerance=1e-4,
                       init=None, \
                       dump=False, dump_path=None, random_state=21, verbose=False, variation='elkan', debug=False,
                       max_iter=300, yinyang_t=0.5, device=1, kmcuda_path=None, \
                       vis_fig=False, vis_points=5, vis_perplexity=30, vis_iter=2500, global_step=-1, tensorboard=None):
    if lib == 'sklearn':
        if init is None:
            sklearn_init = 'k-means++'
        else:
            sklearn_init = init

        if debug:
            print("Inside clustering")
            print("Number of clusters: ", str(num_clusters))

        kmeans_model = KMeans(init=sklearn_init, n_clusters=num_clusters, n_jobs=cores, \
                              verbose=verbose, algorithm=variation, random_state=random_state, tol=tolerance,
                              max_iter=max_iter)

        kmeans_model.fit(data_vec)
        cluster_labels = kmeans_model.labels_
        cluster_to_words = find_word_clusters(words, cluster_labels)

        if debug:
            print("Number of clusters: ", str(len(cluster_to_words)))

        if len(cluster_to_words) < num_clusters:
            cluster_to_words, empty_cluster_ids = handle_empty_clusters_better(cluster_to_words, num_clusters)
            print("Number of clusters after handling empty clusters : ", str(len(cluster_to_words)))
            print("empty_cluster_ids are ", empty_cluster_ids)
        cluster_center_vectors = kmeans_model.cluster_centers_

    elif lib == 'kmcuda':
        # Just support init = kmeans++
        if kmcuda_path is not None:
            sys.path.append(kmcuda_path)
        else:
            print("Please provide path to kmcuda!")

        from libKMCUDA import kmeans_cuda
        np.random.seed(random_state)
        if metric == 'euclidean':
            cluster_center_vectors, assignments = kmeans_cuda(data_vec, num_clusters, tolerance=tolerance, verbosity=1,
                                                              seed=random_state, device=device, metric="L2",
                                                              yinyang_t=yinyang_t)
        elif metric == 'cosine':
            data_vec_normalized = data_vec / (np.linalg.norm(data_vec, axis=1).reshape(-1, 1))
            cluster_center_vectors, assignments = kmeans_cuda(data_vec_normalized, num_clusters, tolerance=tolerance,
                                                              verbosity=1, seed=random_state, device=device,
                                                              metric="cos", yinyang_t=yinyang_t)

        cluster_to_words = find_word_clusters(words, assignments)

        if len(cluster_to_words) < num_clusters:
            cluster_to_words, empty_cluster_ids = handle_empty_clusters_better(cluster_to_words, num_clusters)
            print("Number of clusters after handling empty clusters : ", str(len(cluster_to_words)))
            print("empty_cluster_ids are ", empty_cluster_ids)

    else:
        print("Not implemented error: ", lib)
        cluster_to_words = None

    if vis_fig or dump:
        word_vector_dict = dict(zip(words, data_vec))

    if vis_fig:
        vis_time_st = time.perf_counter()
        visualize_clustering(cluster_to_words, word_vector_dict, cluster_center_vectors, points_per_cluster=vis_points, \
                             perplexity=vis_perplexity, n_iter=vis_iter, random_state=random_state,
                             global_step=global_step, tensorboard=tensorboard)
        vis_time_end = time.perf_counter()
        logging.info("Time taken in visualize_clustering:  " + str(vis_time_end - vis_time_st) + " seconds")

    if dump and (dump_path is not None):
        clustering_info = str(num_clusters) + "_" + lib + "_" + metric + "_" + variation + "_" + str(
            tolerance) + "_" + str(random_state) + "_" + utils.get_timestamp()
        subdir_name = os.path.join(dump_path, clustering_info)
        utils.mkdir(subdir_name)
        word_vector_pkl = open(os.path.join(subdir_name, "word_vector.pickle"), "wb")
        cluster_center_pkl = open(os.path.join(subdir_name, "cluster_center.pickle"), "wb")
        cluster_to_words_pkl = open(os.path.join(subdir_name, "cluster_to_words.pickle"), "wb")

        print("Pickling word_vector_dict")
        pickle.dump(word_vector_dict, word_vector_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickling cluster_center_vectors")
        pickle.dump(cluster_center_vectors, cluster_center_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickling cluster_to_words")
        pickle.dump(cluster_to_words, cluster_to_words_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return cluster_to_words


def main(args):
    params = utils.dotdict(vars(args))
    print('-------- PARAMETERS --------')
    print(json.dumps(params, sort_keys=True, indent=4))
    print('----------------------------')

    logging.basicConfig()
    if params.log:
        level = logging.getLevelName(params.log)
        logging.getLogger().setLevel(level)

    logging.info("Building word_vector_matrix")
    if params.vocab_file != '':
        logging.info("Loading vocabulary")
        opts = {}
        opts['vocab_fpath'] = params.vocab_file
        analysis = Analysis(opts=opts)
        analysis.get_vocab_counts = True
        analysis.set_id2word_cooc()
        vocab = analysis.id2word_cooc
        word2id = {word_ct_pair[0]: idx for idx, word_ct_pair in enumerate(vocab)}
        print(len(word2id), "the len of word2id is ")
        params.n_words = len(vocab)
        df, labels_array = build_word_vector_matrix(params.input_vector_file, params.n_words, params.dim, word2id,
                                                    context_vectors=params.context_vectors, \
                                                    no_lower=params.no_lower)
        print(len(df), "the len of word vectors passed to clustering is ")
        logging.info("#Words with vectors present " + str(len(labels_array)))
    else:
        df, labels_array = build_word_vector_matrix(params.input_vector_file, params.n_words, params.dim,
                                                    context_vectors=params.context_vectors)

    st_time = time.perf_counter()
    if params.lib == 'sklearn':
        if params.algo == 'kmeans':
            # use all but one cores available on system
            logging.info("Using Kmeans")
            if params.init_path == "":
                kmeans_model = KMeans(init='k-means++', n_clusters=params.clusters_to_make, n_jobs=params.num_cores, \
                                      verbose=params.verbose, algorithm=params.variation,
                                      random_state=params.random_state, max_iter=params.max_iter, tol=params.tol)
        elif params.algo == 'minibatchkmeans':
            logging.info("Using MiniBatch Kmeans")
            kmeans_model = MiniBatchKMeans(init='k-means++', n_clusters=params.clusters_to_make)

        kmeans_model.fit(df)
        logging.info("Clustering done")

        cluster_labels = kmeans_model.labels_
        # cluster_to_words: { '0': ['capt', 'restated', ...], ... }
        cluster_to_words = find_word_clusters(labels_array, cluster_labels)
        cluster_center_vectors = kmeans_model.cluster_centers_

    elif params.lib == 'kmcuda':
        kmcuda_path = "/mlodata1/sidak/projects/kmcuda/src/"
        sys.path.append(kmcuda_path)
        from libKMCUDA import kmeans_cuda
        np.random.seed(params.random_state)
        df = df.astype(np.float32)
        if params.metric == 'euclidean':
            cluster_center_vectors, assignments = kmeans_cuda(df, params.clusters_to_make, tolerance=params.tol,
                                                              verbosity=1, seed=params.random_state,
                                                              device=params.device)
        elif params.metric == 'cosine':
            df_normalized = df / (np.linalg.norm(df, axis=1).reshape(-1, 1))
            cluster_center_vectors, assignments = kmeans_cuda(df_normalized, params.clusters_to_make,
                                                              tolerance=params.tol, verbosity=1,
                                                              seed=params.random_state, device=params.device,
                                                              metric="cos", yinyang_t=params.yinyang)

        cluster_to_words = find_word_clusters(labels_array, assignments)

    end_time = time.perf_counter()
    logging.info("Time taken in clustering:  " + str(end_time - st_time) + " seconds")

    word_vector_dict = dict(zip(labels_array, df))

    logging.info("Dumping clustering results")

    utils.mkdir(params.target_folder)

    clustering_info = params.algo + "_" + str(params.clusters_to_make) + "_" + str(
        params.n_words) + "_" + params.lib + "_" + params.metric + "_" + str(params.device) + "_" + str(params.yinyang)

    if params.nick != "":
        clustering_info += "_" + params.nick

    subdir_name = os.path.join(params.target_folder, clustering_info)
    utils.mkdir(subdir_name)

    word_vector_pkl = open(os.path.join(subdir_name, "word_vector.pickle"), "wb")
    cluster_center_pkl = open(os.path.join(subdir_name, "cluster_center.pickle"), "wb")
    cluster_to_words_pkl = open(os.path.join(subdir_name, "cluster_to_words.pickle"), "wb")

    logging.info("Pickling word_vector_dict")
    pickle.dump(word_vector_dict, word_vector_pkl, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Pickling cluster_center_vectors")
    pickle.dump(cluster_center_vectors, cluster_center_pkl, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Pickling cluster_to_words")
    pickle.dump(cluster_to_words, cluster_to_words_pkl, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering word vectors.')
    parser.add_argument('--input-vector-file', action='store', type=str, required=True,
                        help='path to GloVe/Word2Vec vectors')
    parser.add_argument('--algo', action='store', type=str, default="kmeans", required=True,
                        help='Clustering algo to use. Possibilities: kmeans, minibatchkmeans')
    parser.add_argument('--clusters-to-make', action='store', default=100, type=int, required=True,
                        help='Number of clusters to make (default: 100)')
    parser.add_argument('--n-words', action='store', default=205704, type=int, required=False,
                        help='Size of vocabulary  (Possibilities: 205704, 205513, |V|)')
    parser.add_argument('--vocab-file', action='store', type=str, required=False, default='',
                        help='path to vocabulary')
    parser.add_argument('--lib', action='store', type=str, default="sklearn", required=False,
                        help='The variation of kmeans clustering algo to use. Possibilities: sklearn, kmcuda')
    parser.add_argument('--metric', action='store', type=str, default="euclidean", required=False,
                        help='The variation of kmeans clustering algo to use. Possibilities: euclidean, cosine')
    parser.add_argument('--yinyang', action='store', default=0.5, type=float,
                        help='Yinyang_t value for KMCUDA (default: 0.5)')
    parser.add_argument('--device', action='store', default=-1, type=int,
                        help='Clustering device id for KMCUDA. Remember it is bitwise or of the devices you want to use. (default: 1 [first device])')
    parser.add_argument('--variation', action='store', type=str, default="elkan", required=False,
                        help='The variation of kmeans clustering algo to use. Possibilities: elkan, full')
    parser.add_argument('--dim', action='store', default=300, type=int, required=False,
                        help='Number of dimensions of the vector (default: 300)')
    parser.add_argument('--tol', action='store', default=1e-4, type=float, required=False,
                        help='Number of dimensions of the vector (default: 1e-4)')
    parser.add_argument('--max-iter', action='store', default=300, type=int, required=False,
                        help='Number of dimensions of the vector (default: 300)')
    parser.add_argument('--random-state', action='store', default=21, type=int, required=False,
                        help='Random State for kmeans (default: 21)')
    parser.add_argument('--target-folder', action='store', type=str, required=True,
                        help='path to target folder to save the cluster results')
    parser.add_argument('--init-path', action='store', type=str, required=False, default="",
                        help='path for initial choice of centroids')
    parser.add_argument('--verbose', action='store_true',
                        help='Perform KMeans in the verbose mode')
    parser.add_argument('--log', action='store', type=str, default="INFO",
                        help='log level')
    parser.add_argument('--context-vectors', action='store_true',
                        help='Extract context vectors instead')
    parser.add_argument('--nick', action='store', type=str, default="",
                        help='Nickname for the launched experiment')
    parser.add_argument('--num-cores', action='store', default=10, type=int,
                        help='Number of CPU cores for kmeans clustering (default: 10)')
    parser.add_argument('--no-lower', action='store_true',
                        help='Dont lower case words while extracting vector info')
    args = parser.parse_args()
    main(args)
