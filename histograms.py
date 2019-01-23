import pickle
import os
from cooc_analysis import Analysis
import argparse
import utils
import json
import logging
import math
import numpy as np
import time
import scipy.sparse as sp
import flags

'''
Usage
    python histograms.py 
    	--cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ 
        --ppmi-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ppmi_smooth_0.75_k-shift_1.0.npz 
        --cluster-data-dir ./data/vectors/clusters/book_glove_symmetric=1_window-size=10_min-count=10_eta=0.005_iter=75_cleaned=300/kmeans_100_205513
		--histogram-data-dir ./data/histograms/
'''


def compute_cluster_histograms(word2id, cluster_to_words, ppmi_path, painful_words, dump=False, dump_path=None,
                               debug=False, num_bins_ideally=100, ppmi_bin_norm=False, bin_norm_exp=1.0):
    # To prevent side effects due to mutability
    _cluster_to_words = dict(cluster_to_words)
    num_clusters = len(_cluster_to_words)
    num_words = len(word2id)
    desired_cols = list(_cluster_to_words.keys())
    print("In histograms.py")
    print("Number of clusters ideally is ", str(num_bins_ideally))
    print("Number of clusters actually is ", str(len(_cluster_to_words)))
    print("Number of words is ", str(num_words))

    if (len(painful_words) != 0):
        painful_cluster_id = str(num_bins_ideally)
        _cluster_to_words[painful_cluster_id] = painful_words
        num_clusters += 1
        print("cluster keys after ", list(_cluster_to_words.keys()))

        print(len(_cluster_to_words), "len of cluster_to_words", str(painful_cluster_id))
        print("Number of clusters ideally after handling painful words is ", str(len(_cluster_to_words)))
    # print(_cluster_to_words)
    else:
        print("No painful words found in compute_cluster_histograms")

    cluster_rows = []
    cluster_cols = []
    cluster_data = []

    for cluster_id in _cluster_to_words:
        words_in_cluster = _cluster_to_words[cluster_id]
        cluster_key = int(cluster_id)
        cluster_data += [1] * len(words_in_cluster)
        cluster_rows += [cluster_key] * len(words_in_cluster)
        cluster_ids = [word2id[wd] for wd in words_in_cluster]
        cluster_cols += cluster_ids

    if debug:
        print("DEBUGGING HISTS")
        print("len of _cluster_to_words ", len(_cluster_to_words))
        print("len of cluster_rows ", len(cluster_rows))
        print("len of cluster_cols ", len(cluster_cols))
        print("len of cluster_data ", len(cluster_data))
        print("num of clusters actually is ", len(_cluster_to_words))
        print("num of clusters ideally is ", num_bins_ideally)

    sps_cluster = sp.csr_matrix((cluster_data, (cluster_rows, cluster_cols)), shape=(num_clusters, num_words))
    ppmi_sps_matrix = sp.load_npz(os.path.join(ppmi_path))
    sps_hist = ppmi_sps_matrix * sps_cluster.T

    if (len(painful_words) != 0):
        print("Print sps_hist shape before" + str(sps_hist.shape))
        sps_hist = sps_hist[:, desired_cols]
        print("Print sps_hist shape after" + str(sps_hist.shape))

    ## Perform Bin Normalization for learning mode!
    if ppmi_bin_norm:
        col_marginals = sps_hist.sum(axis=0)
        col_marginals_np = np.asarray(col_marginals)
        print("Min, Median, Mean & Max values of column ppmi_cluster marginals")
        print(np.min(col_marginals_np), np.median(col_marginals_np), np.mean(col_marginals_np), \
              np.max(col_marginals_np))

        if bin_norm_exp < 1:
            col_marginals = np.power(col_marginals, bin_norm_exp)

        col_marginals = col_marginals.reshape((1, -1))
        sps_hist = sps_hist.multiply(1.0 / col_marginals)
        print("Done PPMI bin_normalization. Verifying:")
        col_marginals_np_again = np.asarray(sps_hist.sum(axis=0))
        print("Min, Median, Mean & Max values of the bin normalized column ppmi_cluster marginals")
        print(np.min(col_marginals_np_again), np.median(col_marginals_np_again), np.mean(col_marginals_np_again), \
              np.max(col_marginals_np_again))

    sps_hist_marginals = sps_hist.sum(axis=1)
    sps_hist_normalized = sps_hist.multiply(1.0 / sps_hist_marginals)
    dense_hist = sps_hist_normalized.todense()
    if dump and (dump_path is not None):
        ppmi_version = ppmi_path.split('/')[-1].replace('.npz', '')
        histogram_version = ppmi_version + "_" + "num_clusters_" + str(num_clusters) + "_" + utils.get_timestamp()
        histogram_dump_dir = os.path.join(dump_path, histogram_version)
        utils.mkdir(histogram_dump_dir)
        np.savez(os.path.join(histogram_dump_dir, "normalized_cluster_hists.npz"), dense_hist)
    return dense_hist


def main(args):
    params = utils.dotdict(vars(args))
    print('-------- PARAMETERS --------')
    print(json.dumps(params, sort_keys=True, indent=4))
    print('----------------------------')
    logging.basicConfig()

    if params.log:
        level = logging.getLevelName(params.log)
        logging.getLogger().setLevel(level)

    st_time = time.perf_counter()
    word_vector_dict = pickle.load(open(os.path.join(params.cluster_data_dir, 'word_vector.pickle'), 'rb'))
    cluster_center_vectors = pickle.load(open(os.path.join(params.cluster_data_dir, 'cluster_center.pickle'), 'rb'))
    cluster_to_words = pickle.load(open(os.path.join(params.cluster_data_dir, 'cluster_to_words.pickle'), 'rb'))
    end_time = time.perf_counter()
    logging.info("Loaded cluster files !" + " in " + str(end_time - st_time) + " seconds")

    if params.bin_normalization == "num":
        cluster_sizes = []
        for cl in cluster_to_words:
            cluster_sizes.append(len(cluster_to_words[cl]))
        cluster_sizes = np.asarray(cluster_sizes)

    opts = {}
    analysis = Analysis(opts=opts)
    analysis.load_mode = flags.LOAD_MIN
    analysis.setup_cooccurr_analysis(params.cooc_root_path)
    word2id = {word: idx for idx, word in enumerate(analysis.id2word_cooc)}

    unassigned_words = []
    for wd in word2id:
        if wd not in word_vector_dict:
            unassigned_words.append(wd)
    logging.info("Number of words unassigned to any cluster are: " + str(len(unassigned_words)))

    num_clusters = len(cluster_center_vectors)
    num_words = len(word2id)

    if (len(unassigned_words) != 0):
        unassigned_cluster_id = str(num_clusters)
        cluster_to_words[unassigned_cluster_id] = unassigned_words
        num_clusters += 1
        print(len(cluster_to_words), "len of cluster_to_words", str(unassigned_cluster_id))

    st_time = time.perf_counter()
    idx = 0
    cluster_rows = []
    cluster_cols = []
    cluster_data = []
    net_mean_dist = 0.0
    net_stddev_dist = 0.0
    net_med_dist = 0.0
    net_max_dist = 0.0
    net_min_dist = 0.0
    net_ct = 0
    for cluster_id in cluster_to_words:
        words_in_cluster = cluster_to_words[cluster_id]
        cluster_key = int(cluster_id)
        if params.distance_weighting and (cluster_id != unassigned_cluster_id):
            word_vec_arr = [word_vector_dict[wd] for wd in words_in_cluster]
            word_vec_arr = np.asarray(word_vec_arr)
            word_vec_arr -= cluster_center_vectors[int(cluster_id)]
            distance_to_center = np.linalg.norm(word_vec_arr, axis=1)
            mean_dist = np.mean(distance_to_center)
            stddev_dist = np.std(distance_to_center)
            outlier_dist = mean_dist + params.num_sigma * stddev_dist
            distance_wts = np.clip(outlier_dist / distance_to_center, a_min=0.0, a_max=1.0)
            cluster_data += distance_wts.tolist()
            net_mean_dist += mean_dist
            net_stddev_dist += stddev_dist
            net_med_dist += np.median(distance_to_center)
            net_max_dist += np.max(distance_to_center)
            net_min_dist += np.min(distance_to_center)
            net_ct += 1
        else:
            cluster_data += [1] * len(words_in_cluster)  # add maybe a weight here if you want

        cluster_rows += [cluster_key] * len(words_in_cluster)
        cluster_cols += [word2id[wd] for wd in words_in_cluster]

    if params.distance_weighting:
        print("Overall mean_dist across clusters: ", str(net_mean_dist / net_ct))
        print("Overall stddev_dist across clusters: ", str(net_stddev_dist / net_ct))
        print("Overall med_dist across clusters: ", str(net_med_dist / net_ct))
        print("Overall max_dist across clusters: ", str(net_max_dist / net_ct))
        print("Overall min_dist across clusters: ", str(net_min_dist / net_ct))

    sps_cluster = sp.csr_matrix((cluster_data, (cluster_rows, cluster_cols)), shape=(num_clusters, num_words))
    end_time = time.perf_counter()
    logging.info("Created cluster sps matrix !" + " in " + str(end_time - st_time) + " seconds")

    st_time = time.perf_counter()
    ppmi_sps_matrix = sp.load_npz(os.path.join(params.ppmi_path))
    end_time = time.perf_counter()
    logging.info("Loaded ppmi sps matrix !" + " in " + str(end_time - st_time) + " seconds")

    st_time = time.perf_counter()
    sps_hist = ppmi_sps_matrix * sps_cluster.T
    end_time = time.perf_counter()
    logging.info("Computed cluster hists !" + " in " + str(end_time - st_time) + " seconds")

    if (len(unassigned_words) != 0):
        # get rid of the last column and also later the rows which don't have corresponding word vectors
        logging.info("Print sps_hist shape before" + str(sps_hist.shape))
        sps_hist = sps_hist[:, 0:-1]
        logging.info("Print sps_hist shape after" + str(sps_hist.shape))

    if params.bin_normalization == "ppmi":
        col_marginals = sps_hist.sum(axis=0)
        col_marginals_np = np.asarray(col_marginals)
        print("Min, Median, Mean & Max values of column ppmi_cluster marginals")
        print(np.min(col_marginals_np), np.median(col_marginals_np), np.mean(col_marginals_np), \
              np.max(col_marginals_np))

        if params.bin_norm_exp < 1:
            col_marginals = np.power(col_marginals, params.bin_norm_exp)

        col_marginals = col_marginals.reshape((1, -1))
        sps_hist = sps_hist.multiply(1.0 / col_marginals)
        print("Done PPMI bin_normalization. Verifying:")
        col_marginals_np_again = np.asarray(sps_hist.sum(axis=0))
        print("Min, Median, Mean & Max values of the bin normalized column ppmi_cluster marginals")
        print(np.min(col_marginals_np_again), np.median(col_marginals_np_again), np.mean(col_marginals_np_again), \
              np.max(col_marginals_np_again))

    elif params.bin_normalization == "num":

        col_marginals_np = np.asarray(sps_hist.sum(axis=0))
        print("Min, Median, Mean & Max values of column ppmi_cluster marginals")
        print(np.min(col_marginals_np), np.median(col_marginals_np), np.mean(col_marginals_np), \
              np.max(col_marginals_np))
        sps_hist = sps_hist.multiply(1.0 / cluster_sizes)
        print("Done NUM bin_normalization. Verifying:")
        col_marginals_np_again = np.asarray(sps_hist.sum(axis=0))
        print("Min, Median, Mean & Max values of the bin normalized column ppmi_cluster marginals")
        print(np.min(col_marginals_np_again), np.median(col_marginals_np_again), np.mean(col_marginals_np_again), \
              np.max(col_marginals_np_again))
    else:
        print("not doing any bin normalization")

    st_time = time.perf_counter()
    sps_hist_marginals = sps_hist.sum(axis=1)
    sps_hist_normalized = sps_hist.multiply(1.0 / sps_hist_marginals)
    dense_hist = sps_hist_normalized.todense()
    end_time = time.perf_counter()
    logging.info("Computed dense normalized hists !" + " in " + str(end_time - st_time) + " seconds")

    st_time = time.perf_counter()

    clustering_version = params.cluster_data_dir.split('/')[-1]
    ppmi_version = params.ppmi_path.split('/')[-1].replace('.npz', '')
    histogram_version = ppmi_version + "_" + clustering_version
    if params.nick != "":
        histogram_version += "_" + params.nick
    if params.bin_normalization != None:
        histogram_version += "_" + params.bin_normalization + "_bin_norm"
        histogram_version += "_" + str(params.bin_norm_exp)

    histogram_dump_dir = os.path.join(params.histogram_data_dir, histogram_version)
    utils.mkdir(histogram_dump_dir)

    np.savez(os.path.join(histogram_dump_dir, "normalized_cluster_hists.npz"), dense_hist)
    end_time = time.perf_counter()
    logging.info("Dump dense normalized hists !" + " in " + str(end_time - st_time) + " seconds")
    # logging.info("Dump dir is ", str(histogram_dump_dir))
    logging.info("-------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Cluster Histograms')
    parser.add_argument('--cooc-root-path', action='store', type=str, required=True,
                        help='path to folder containing cooc pickled files')
    parser.add_argument('--ppmi-path', action='store', type=str, required=True,
                        help='path to ppmi file to use')
    parser.add_argument('--cluster-data-dir', action='store', type=str, required=True,
                        help='path to folder containing cluster data files')
    parser.add_argument('--histogram-data-dir', action='store', type=str, required=True,
                        help='path to folder containing histogram data files')
    parser.add_argument('--nick', action='store', type=str, required=False, default="",
                        help='path to folder containing histogram data files')
    parser.add_argument('--distance-weighting', action='store_true',
                        help='Weight contributions to cluster based on distance to cluster center')
    parser.add_argument('--num-sigma', action='store', type=float, default=1.0,
                        help='Number of sigmas beyond which to consider outlier')
    parser.add_argument('--log', action='store', type=str, default="INFO",
                        help='log level')
    parser.add_argument('--bin-normalization', type=str, default=None,
                        help='If the cluster bins should be normalized. \
                        Possibilities: (ppmi) PPMI Cluster marginals, (size) Cluster size')
    parser.add_argument('--bin-norm-exp', action='store', type=float, default=1.0,
                        help='Exponent of bin normalization. \
                        Possibilities: (ppmi) PPMI Cluster marginals, (size) Cluster size')
    args = parser.parse_args()
    main(args)
