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
from sklearn.metrics.pairwise import euclidean_distances
import ot
from sklearn.decomposition import TruncatedSVD

from scipy.special import expit as sigmoid

'''
Usage
	python wasserstein.py 
		--cluster-data-dir ./data/vectors/clusters/book_glove_symmetric=1_window-size=10_min-count=10_eta=0.005_iter=75_cleaned=300/kmeans_100_205513
		--cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ 
		--hists-path ./data/histograms/ppmi_smooth_0.75_k-shift_1.0_kmeans_100_205513/normalized_cluster_hists.npz
		--word1 cute --word2 animal
		--marginals-path ./data/cooc/book_coocurrence_symmetric=1_window-size=10_cleaned=300/ppmi_smooth_0.75_k-shift_1.0_marginals.npz
'''


# -------------------------- Adapted from SIF ------------------#
def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    print(X.shape, "shape of test embedding matrix")
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


# -------------------------- Adapted from SIF ------------------#

def cosine_sim(vec1, vec2):
    EPSILON = 1e-7
    vec1 += EPSILON * np.ones(len(vec1))
    vec2 += EPSILON * np.ones(len(vec1))
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_angular_ground_metric(vectors):
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    cos_sim = normalized_vectors @ normalized_vectors.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angular_gm = np.arccos(cos_sim)
    return angular_gm


def get_wass_cosine(hist1, hist2, cluster_center_vectors):
    vec1 = get_weighted_wass_emb(hist1, cluster_center_vectors)
    vec2 = get_weighted_wass_emb(hist2, cluster_center_vectors)
    return cosine_sim(vec1, vec2)


def get_wass_euclid(hist1, hist2, cluster_center_vectors, normalize=False):
    vec1 = get_weighted_wass_emb(hist1, cluster_center_vectors)
    vec2 = get_weighted_wass_emb(hist2, cluster_center_vectors)
    if normalize:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
    return np.linalg.norm(vec1 - vec2)


def get_softsim(hist1, hist2, ground_metric_matrix):
    total_sim = 0.0
    for i in range(len(hist1)):
        for j in range(len(hist2)):
            total_sim += (hist1[i] * hist2[j] * (1 - ground_metric_matrix[i, j]))
    return total_sim


def get_weighted_wass_emb(hist, cluster_center_vectors):
    vec = np.zeros(cluster_center_vectors.shape[1])
    for i in range(len(hist)):
        vec += hist[i] * cluster_center_vectors[i, :]
    return vec


def get_wass_dirac_dist(word_vec, other_hist, cluster_center_vectors, metric='euclidean', wass2=False):
    dist = 0.0
    if metric == 'cosine':
        for i in range(len(other_hist)):
            dist += other_hist[i] * (1 - cosine_sim(word_vec, cluster_center_vectors[i, :]))
    elif metric == 'euclidean':
        for i in range(len(other_hist)):
            if not wass2:
                dist += other_hist[i] * np.linalg.norm(word_vec - cluster_center_vectors[i, :])
            else:
                dist += other_hist[i] * (np.linalg.norm(word_vec - cluster_center_vectors[i, :]) ** 2)
    return dist


def get_wass_bary_fast(word1, word2, cluster_histograms_, word2id, ground_metric, reg=0.1, alpha=0.5, normalize=None):
    print("CHECK: IN WASS BARY FAST")
    if normalize == 'max':
        print("Doing max normalization")
        ground_metric = ground_metric / ground_metric.max()
    weights = np.array([1 - alpha, alpha])
    # print(cluster_histograms_.shape)
    normalized_cluster_hist1 = cluster_histograms_[word2id[word1]]
    normalized_cluster_hist2 = cluster_histograms_[word2id[word2]]
    wass_hists = np.vstack((normalized_cluster_hist1, normalized_cluster_hist2)).T
    bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, reg, weights)
    return bary_wass


# we can try two things: alpha uniform and the other case proportional to the ppmi value or tf-idf

def get_wass_bary_fast_mult(words, cluster_histograms_, word2id, ground_metric, mode="plain", \
                            reg=0.1, alpha='uniform', weights=None, marginals=None, smoothing=1.0,
                            length_correction=False, seq=False, \
                            max_its=1000):
    if mode == "btree":
        if marginals is None:
            print("Error: Marginals not found")
            return None
        else:
            return get_wass_bary_btree(words, cluster_histograms_, word2id, ground_metric, \
                                       marginals, reg, alpha, smoothing, length_correction, seq)

    elif mode == 'half':
        return get_wass_bary_in_halfs(words, cluster_histograms_, word2id, ground_metric, \
                                      reg=reg, marginals=marginals, smoothing=smoothing,
                                      length_correction=length_correction, max_its=max_its)
    else:

        if weights is None and alpha == 'uniform':
            weights = np.ones(len(words)) / len(words)
        else:
            # print("using ", alpha)
            weights = weights

        hists_list = []
        for wd in words:
            hists_list.append(cluster_histograms_[word2id[wd]])

        wass_hists = np.vstack(hists_list).T
        bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, reg, weights, numItermax=max_its)
        return bary_wass


def get_interpolated_wass_bary_fast_mult(words, cluster_histograms_, word2id, wordvecs, clustervecs, weights, args):
    hists_list = []

    for idx, wd in enumerate(words):
        # make a zeros histogram of size num_words + num_clusters
        mod_hist = np.zeros(cluster_histograms_[word2id[wd]].shape[0] + len(words))
        mod_hist[idx] = args.interpolate_repr
        mod_hist[len(words):] = (1 - args.interpolate_repr) * cluster_histograms_[word2id[wd]]
        hists_list.append(mod_hist)

    wass_hists = np.vstack(hists_list).T

    coord = np.vstack([wordvecs, clustervecs])
    ground_metric = obtain_ground_metric(args, coord)

    bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, args.regbary, weights, numItermax=args.max_its)

    return bary_wass


def get_harmonic_mean(a, b):
    ans = 2.0 * a
    ans /= (a + b)
    ans *= b
    return ans


def get_geometric_mean(a, b):
    return math.sqrt(a * b)


def get_wass_bary_btree(words, cluster_histograms_, word2id, ground_metric, \
                        marginals, reg=0.1, alpha='hm_cooc_btree', smoothing=1.0, length_correction=False, seq=False):
    marginals = np.power(marginals, smoothing)
    left_idx = 0
    right_idx = len(words)
    word_hists = [cluster_histograms_[word2id[wd]] for wd in words]
    word_margs = [marginals[word2id[wd]] for wd in words]

    if not seq:
        bary, _ = compute_bary_recursive(left_idx, right_idx, word_hists, word_margs, ground_metric, reg, alpha,
                                         length_correction)
    else:
        bary, _ = compute_bary_sequential(left_idx, right_idx, word_hists, word_margs, ground_metric, reg, alpha,
                                          length_correction)

    print("Ultimate level: For words between ", left_idx, " and ", right_idx, " the marginal is ", _)
    return bary


def compute_bary_recursive(left_idx, right_idx, word_hists, word_margs, ground_metric, reg, alpha, a=1e-3,
                           length_correction=False):
    # [left_idx, right_idx)
    # i.e., right_idx is not included

    if (right_idx - left_idx) == 1:
        return word_hists[left_idx], word_margs[left_idx]

    mid_idx = int((left_idx + right_idx) / 2)

    left_bary, left_marginal = compute_bary_recursive(left_idx, mid_idx, word_hists, word_margs, ground_metric, reg,
                                                      alpha)
    print("For words between ", left_idx, " and ", mid_idx, " the marginal is ", left_marginal)
    right_bary, right_marginal = compute_bary_recursive(mid_idx, right_idx, word_hists, word_margs, ground_metric, reg,
                                                        alpha)
    print("For words between ", mid_idx, " and ", right_idx, " the marginal is ", right_marginal)
    wts = np.zeros(2)
    wts[0] = 1 / (a + left_marginal)
    wts[1] = 1 / (a + right_marginal)
    if length_correction:
        wts[0] /= (mid_idx - left_idx) ** 3.0
        wts[1] /= (right_idx - mid_idx) ** 3.0
    print("For words between ", left_idx, " and ", right_idx, " the wt are ", wts)
    normalized_wts = wts / wts.sum()
    print("For words between ", left_idx, " and ", right_idx, " the normalized_wts are ", normalized_wts)
    if alpha == "am_cooc_btree":
        net_marginal = (left_marginal + right_marginal) / 2.0
    elif alpha == "gm_cooc_btree":
        print(" get_geometric_mean")
        net_marginal = get_geometric_mean(left_marginal, right_marginal)
    else:
        print(" get_harmonic_mean")
        net_marginal = get_harmonic_mean(left_marginal, right_marginal)

    wass_hists = np.vstack((left_bary, right_bary)).T
    bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, reg, normalized_wts)
    return bary_wass, net_marginal


def compute_bary_recursive_kway(k, left_idx, right_idx, word_hists, word_margs, ground_metric, reg, alpha, a=1e-3,
                                length_correction=False):
    if (right_idx - left_idx) == 1:
        return word_hists[left_idx], word_margs[left_idx]

    child_barycenters = []
    child_marginals = []
    child_lens = []

    slice_len = int(math.ceil((right_idx - left_idx) / k))

    for slice_idx in range(left_idx, right_idx, ):
        bary, marg = compute_bary_recursive_kway(k, slice_idx, max(slice_idx + k, right_idx), word_hists, \
                                                 word_margs, ground_metric, reg, alpha, a, length_correction)
        print("For words between ", slice_idx, " and ", max(slice_idx + k, right_idx), " the marginal is ", marg)
        child_barycenters.append(bary)
        child_marginals.append(marg)
        child_lens.append(max(slice_idx + k, right_idx) - slice_idx)

    wts = np.zeros(len(child_marginals))
    for idx, child_marg in enumerate(child_marginals):
        # no need for a/a+child_marg as the one in the numerator will be cancelled
        wts[idx] = 1 / (a + child_marg)
        if length_correction:
            wts[idx] /= child_lens[idx]

        print("For words between ", idx * k, " and ", max((idx + 1) * k, right_idx), " the overall wt are ", wts[idx])

    normalized_wts = wts / wts.sum()

    for idx, child_wt in enumerate(normalized_wts):
        print("For words between ", idx * k, " and ", max((idx + 1) * k, right_idx), " the overall normalized_wts are ",
              normalized_wts[idx])

    if alpha == "am_cooc_btree":
        net_marginal = np.mean(child_marginals)
    elif alpha == "gm_cooc_btree":
        print(" get_geometric_mean")
        net_marginal = get_geometric_mean(left_marginal, right_marginal)
    else:
        print(" get_harmonic_mean")
        net_marginal = get_harmonic_mean(left_marginal, right_marginal)

    wass_hists = np.vstack(child_barycenters).T
    bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, reg, normalized_wts)
    return bary_wass, net_marginal


def get_wass_bary_in_halfs(words, cluster_histograms_, word2id, ground_metric, \
                           reg=0.1, marginals=None, smoothing=1.0, length_correction=False, max_its=1000, a=1e-3):
    print("len of words is ", len(words))
    mid_idx = int(len(words) / 2)

    left_wts = get_wts(marginals, word2id, words[0:mid_idx], smoothing, a, return_normalized=False)
    right_wts = get_wts(marginals, word2id, words[mid_idx:len(words)], smoothing, a, return_normalized=False)

    left_marginal = left_wts.sum()
    right_marginal = right_wts.sum()
    print("left_marginal: {} and right_marginal: {}".format(left_marginal, right_marginal))
    left_wts /= left_marginal
    right_wts /= right_marginal

    left_bary = get_wass_bary_fast_mult(words[0:mid_idx], cluster_histograms_, word2id, ground_metric, mode="plain", \
                                        reg=reg, weights=left_wts, max_its=max_its)

    right_bary = get_wass_bary_fast_mult(words[mid_idx:len(words)], cluster_histograms_, word2id, ground_metric,
                                         mode="plain", \
                                         reg=reg, weights=right_wts, max_its=max_its)

    final_wts = [left_marginal, right_marginal]
    final_wts /= (left_marginal + right_marginal)

    wass_hists = np.vstack((left_bary, right_bary)).T
    bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, reg, final_wts)

    return bary_wass


def compute_bary_sequential(left_idx, right_idx, word_hists, word_margs, ground_metric, reg, alpha, a=1e-3,
                            length_correction=False):
    # [left_idx, right_idx)
    # i.e., right_idx is not included

    if (right_idx - left_idx) == 1:
        return word_hists[left_idx], word_margs[left_idx]

    # mid_idx = int((left_idx + right_idx)/2)

    left_bary, left_marginal = compute_bary_sequential(left_idx, right_idx - 1, word_hists, word_margs, ground_metric,
                                                       reg, alpha)
    print("For words between ", left_idx, " and ", right_idx - 1, " the marginal is ", left_marginal)
    right_bary, right_marginal = word_hists[right_idx - 1], word_margs[right_idx - 1]
    print("For words between ", right_idx - 1, " and ", right_idx, " the marginal is ", right_marginal)
    wts = np.zeros(2)

    wts[0] = 1 / ((a + left_marginal) * (right_idx - 1 - left_idx))
    wts[1] = 1 / (a + right_marginal)
    if length_correction:
        wts[0] /= (mid_idx - left_idx) ** 3.0
        wts[1] /= (right_idx - mid_idx) ** 3.0
    print("For words between ", left_idx, " and ", right_idx, " the wt are ", wts)
    normalized_wts = wts / wts.sum()
    print("For words between ", left_idx, " and ", right_idx, " the normalized_wts are ", normalized_wts)
    if alpha == "am_cooc_btree":
        net_marginal = (left_marginal + right_marginal) / 2.0
    elif alpha == "gm_cooc_btree":
        print(" get_geometric_mean")
        net_marginal = get_geometric_mean(left_marginal, right_marginal)
    else:
        print(" get_harmonic_mean")
        net_marginal = get_harmonic_mean(left_marginal, right_marginal)

    wass_hists = np.vstack((left_bary, right_bary)).T
    bary_wass = ot.bregman.barycenter(wass_hists, ground_metric, reg, normalized_wts)
    return bary_wass, net_marginal


def get_wass_dist(hist1, hist2, ground_metric, lambd=0.1, thresh=1e-09, method='sinkhorn_stabilized', \
                  verbose=False, max_its=1000):
    st_time = time.perf_counter()
    ans = ot.sinkhorn2(hist1, hist2, ground_metric, reg=lambd, method=method, stopThr=thresh, verbose=verbose,
                       numItermax=max_its)[0]
    end_time = time.perf_counter()
    logging.info("Computed sinkhorn !" + " in " + str(end_time - st_time) + " seconds")
    return ans


def normalize_ground_metric(args, gm):
    if args.verbose:
        print("Doing normalization: ", args.gm_normalize)

    if args.gm_normalize == 'max':
        return gm / gm.max()
    elif args.gm_normalize == 'median':
        return gm / np.median(gm)
    elif args.gm_normalize == 'mean':
        return gm / gm.mean()
    elif args.gm_normalize == 'log':
        return np.log2(1.0 + gm)
    elif args.gm_normalize == 'none':
        print("no ground_metric NORMALIZATION")
        return gm


def obtain_ground_metric(args, coord, cluster_gm=None):
    # try normalization
    if args.gm_type == 'euclidean':
        if not args.wass2:
            ground_metric_matrix = euclidean_distances(coord)
            print(ground_metric_matrix.max(), "max of ground_metric_matrix")
        else:
            ground_metric_matrix = euclidean_distances(coord)
            ground_metric_matrix = ground_metric_matrix ** 2
            print(ground_metric_matrix.max(), "max of ground_metric_matrix")
            print(ground_metric_matrix.mean(), "mean of ground_metric_matrix")
    elif args.gm_type == 'cosine':
        print("with cosine ground_metric_matrix")
        norms = np.linalg.norm(coord, axis=1)
        normalized_vec = coord / (norms.reshape(-1, 1))
        sim_mat = normalized_vec @ normalized_vec.T
        ground_metric_matrix = 1.0 - sim_mat
        ground_metric_matrix = ground_metric_matrix ** 2
    elif args.gm_type == 'angular':
        ground_metric_matrix = get_angular_ground_metric(coord)

    elif args.gm_type == 'entailment':
        if args.verbose:
            print("Computing entailment ground metric")

        support_size = coord.shape[0]
        ground_metric_matrix = np.zeros((support_size, support_size))

        sub_block_start_ix = support_size  # start index of the subblock that has already been computed

        if cluster_gm is not None:  # in case the ground-metric for the cluster-centers has already been computed
            assert cluster_gm.shape[0] == cluster_gm.shape[1]
            assert cluster_gm.shape[0] == support_size - 2
            ground_metric_matrix[2:, 2:] = cluster_gm
            sub_block_start_ix = 2

        for i in range(support_size):
            for j in range(support_size):
                if i > sub_block_start_ix and j > sub_block_start_ix:
                    continue
                ground_metric_matrix[i, j] = - sigmoid(-coord[i]).dot(np.log(sigmoid(-coord[j])))
        ground_metric_matrix /= 100

        if args.verbose:
            print("entailment_ground_metric.min:", ground_metric_matrix.min())
            print("entailment_ground_metric.max:", ground_metric_matrix.max())

    if args.gm_normalize is not None:
        ground_metric_matrix = normalize_ground_metric(args, ground_metric_matrix)
        if args.debug:
            print("After NORMALIZATION")
            debug_gm_all(np.asarray(ground_metric_matrix), args.regbary)

    if args.clip_gm:
        percent_clipped = (float(
            (ground_metric_matrix >= args.regbary * args.clip_max).sum()) / ground_metric_matrix.size) * 100
        print("inside interpolate: percent_clipped is ", percent_clipped)
        ground_metric_matrix = ground_metric_matrix.clip(min=args.regbary * args.clip_min, \
                                                         max=args.regbary * args.clip_max)

    return ground_metric_matrix


def get_interpolated_wass_dist(hist1, pos1, hist2, pos2, coord, args, cluster_gm=None):
    assert (args.interpolate_repr > 0 and args.interpolate_reduced)
    if args.verbose:
        print("interpolating baby!!!")
    mod_hist1 = np.zeros(hist1.shape[0] + 2)
    mod_hist2 = np.zeros(hist2.shape[0] + 2)
    if args.verbose:
        print("in just interpolated: shape of mod_hist1 is ", mod_hist1.shape)
        print("in just interpolated: shape of mod_hist2 is ", mod_hist2.shape)
    # make 0th index for 1st sentence point repr
    mod_hist1[0] = args.interpolate_repr
    # make 1st index for 1st sentence point repr
    mod_hist2[1] = args.interpolate_repr
    mod_hist1[2:] = (1 - args.interpolate_repr) * hist1
    mod_hist2[2:] = (1 - args.interpolate_repr) * hist2
    mod_coord = np.vstack([pos1, pos2, coord])

    if args.verbose:
        print("shapes of coord and mod_coord are ", coord.shape, mod_coord.shape)
    ground_metric = obtain_ground_metric(args, mod_coord, cluster_gm)

    return get_wass_dist(mod_hist1, mod_hist2, ground_metric, lambd=args.regbary, thresh=args.thresh,
                         max_its=args.max_its)


def get_complete_interpolated_wass_dist(hist1, posns1, hist2, posns2, coord, args):
    assert (args.interpolate_repr > 0 and args.interpolate_complete)
    if args.verbose:
        print("interpolating baby completely!!!")

    assert (hist1.shape[0] + len(posns2)) == (hist2.shape[0] + len(posns1))

    mod_hist1 = np.zeros(hist1.shape[0] + len(posns2))
    mod_hist2 = np.zeros(hist2.shape[0] + len(posns1))
    mod_hist1[0:len(posns1)] = hist1[0:len(posns1)]
    mod_hist1[(len(posns1) + len(posns2)):] = hist1[len(posns1):]

    mod_hist2[len(posns1):(len(posns1) + len(posns2))] = hist2[0:len(posns2)]
    mod_hist2[(len(posns1) + len(posns2)):] = hist2[len(posns2):]

    mod_coord = np.vstack([posns1, posns2, coord])

    ground_metric = obtain_ground_metric(args, mod_coord)

    return get_wass_dist(mod_hist1, mod_hist2, ground_metric, lambd=args.regbary, thresh=args.thresh,
                         max_its=args.max_its)


def get_both_interpolated_wass_dist(hist1, pos1, posns1, hist2, pos2, posns2, other_coords, args):
    assert (args.interpolate_repr > 0 and args.interpolate_complete and args.interpolate_reduced)
    print("interpolating baby really completely!!!")

    assert (hist1.shape[0] + len(posns2)) == (hist2.shape[0] + len(posns1))

    # since hist1 already contains posns1 many places in the front
    mod_hist1 = np.zeros(hist1.shape[0] + len(posns2))
    mod_hist2 = np.zeros(hist2.shape[0] + len(posns1))
    print("number of words in sent1 and 2 are {} and {} resp".format(len(posns1), len(posns2)))
    print("in both interpolated: shape of mod_hist1 is ", mod_hist1.shape)
    print("in both interpolated: shape of mod_hist2 is ", mod_hist2.shape)
    mod_hist1[0:len(posns1)] = hist1[0:len(posns1)]
    mod_hist1[(len(posns1) + len(posns2)):] = hist1[len(posns1):]

    # len(posns1) many zeros before
    mod_hist2[len(posns1):(len(posns1) + len(posns2))] = hist2[0:len(posns2)]
    mod_hist2[(len(posns1) + len(posns2)):] = hist2[len(posns2):]

    mod_coord = np.vstack([posns1, posns2, other_coords])

    return get_interpolated_wass_dist(mod_hist1, pos1, mod_hist2, pos2, mod_coord, args)


def get_barycentric_interpolated_wass_dist(hist1, pos1, hist2, pos2, coord, args):
    print("WARNING! This kind of interpolation is deprecated. Use the one above")
    assert False

    assert (args.interpolate_repr > 0 and args.barycentric_interpolate_reduced)
    print("interpolating baby!!!")
    mod_hist1_a = np.zeros(hist1.shape[0] + 2)
    mod_hist1_b = np.zeros(hist1.shape[0] + 2)

    mod_hist2_a = np.zeros(hist2.shape[0] + 2)
    mod_hist2_b = np.zeros(hist2.shape[0] + 2)
    # make 0th index for 1st sentence point repr
    mod_hist1_a[0] = 1
    mod_hist1_b[2:] = hist1

    mod_hist2_a[1] = 1
    mod_hist2_b[2:] = hist2

    wass_hists_1 = np.vstack((mod_hist1_a, mod_hist1_b)).T
    wass_hists_2 = np.vstack((mod_hist2_a, mod_hist2_b)).T

    mod_coord = np.vstack([pos1, pos2, coord])
    ground_metric = obtain_ground_metric(args, mod_coord)

    hist_wts = np.asarray([args.interpolate_repr, 1 - args.interpolate_repr])

    mod_hist1 = ot.bregman.barycenter(wass_hists_1, ground_metric, args.regbary, hist_wts)

    mod_hist2 = ot.bregman.barycenter(wass_hists_2, ground_metric, args.regbary, hist_wts)

    return get_wass_dist(mod_hist1, mod_hist2, ground_metric, lambd=args.regbary, thresh=args.thresh,
                         max_its=args.max_its)


def get_closest_k_wasserstein_words_fast(query_hist, num_nbrs, interesting_words, cluster_histograms_, ground_metric,
                                         word2id, lambd=0.1, method='sinkhorn_stabilized',
                                         normalize=None):
    print("meta params are", meta_params['data'], meta_params['data']['cluster_data_dir'])
    if (normalize == 'max'):
        ground_metric = ground_metric / ground_metric.max()
    wass_words = []
    wass_dist = []
    for word in interesting_words:
        if word == '\x1d' or word == '\x1cthe':
            continue

        wass_words += [word]

        wass_hist = cluster_histograms_[word2id[word]]
        wass_dist.append(
            ot.sinkhorn2(query_hist, wass_hist, ground_metric, reg=lambd, method=method, verbose='False')[0])
    wass_dist = np.asarray(wass_dist)
    # get indices in ascending order of distance
    sorted_indices = wass_dist.argsort()
    wass_words = np.array(wass_words)
    return wass_words[sorted_indices[:num_nbrs]], wass_dist[sorted_indices[:num_nbrs]]


meta_params = {}


def get_wts(prob_marginal_arr, word2id, words, smoothing=1.0, a=1e-3, return_normalized=True):
    # print("in get_wts marginal a is ", a)
    prob_marginal_arr = np.power(prob_marginal_arr, smoothing)
    marginal_wts = np.zeros(len(words))
    for idx, wd in enumerate(words):
        marginal_wts[idx] = a / (a + prob_marginal_arr[word2id[wd]])

    if return_normalized:
        normalized_wts = marginal_wts / marginal_wts.sum()
        return normalized_wts
    else:
        return marginal_wts


def get_wts_direct(prob_marginal_arr, word2id, words, smoothing=1.0, a=1e-3):
    prob_marginal_arr = np.power(prob_marginal_arr, smoothing)
    marginal_wts = np.zeros(len(words))
    for idx, wd in enumerate(words):
        marginal_wts[idx] = prob_marginal_arr[word2id[wd]]
    normalized_wts = marginal_wts / marginal_wts.sum()
    return normalized_wts


def get_wts_btree(prob_marginal_arr, word2id, words, a=1e-3):
    marginal_wts = np.zeros(len(words))
    for idx, wd in enumerate(words):
        marginal_wts[idx] = a / (a + prob_marginal_arr[word2id[wd]])
    normalized_wts = marginal_wts / marginal_wts.sum()
    return normalized_wts


def main(args):
    params = utils.dotdict(vars(args))
    print('-------- PARAMETERS --------')
    print(json.dumps(params, sort_keys=True, indent=4))
    print('----------------------------')
    logging.basicConfig()
    meta_params['data'] = params
    if params.log:
        level = logging.getLevelName(params.log)
        logging.getLogger().setLevel(level)

    st_time = time.perf_counter()
    word_vector_dict = pickle.load(open(os.path.join(params.cluster_data_dir, 'word_vector.pickle'), 'rb'))
    cluster_center_vectors = pickle.load(open(os.path.join(params.cluster_data_dir, 'cluster_center.pickle'), 'rb'))
    cluster_to_words = pickle.load(open(os.path.join(params.cluster_data_dir, 'cluster_to_words.pickle'), 'rb'))
    end_time = time.perf_counter()
    logging.info("Loaded cluster files !" + " in " + str(end_time - st_time) + " seconds")

    ground_metric_matrix = euclidean_distances(cluster_center_vectors)

    opts = {}
    analysis = Analysis(opts=opts)
    analysis.load_mode = flags.LOAD_MIN
    analysis.setup_cooccurr_analysis(params.cooc_root_path)
    word2id = {word: idx for idx, word in enumerate(analysis.id2word_cooc)}

    st_time = time.perf_counter()
    normalized_cluster_hists = np.load(os.path.join(params.hists_path))['arr_0']
    end_time = time.perf_counter()
    logging.info("Loaded normalized_cluster_hists matrix !" + " in " + str(end_time - st_time) + " seconds")

    marginals_arr = None

    if params.marginals:
        marginals_arr = np.load(params.marginals_path)['arr_0']

    if params.mode == flags.MODE_BARYCENTER_NBRS:

        st_time = time.perf_counter()
        if (params.sent == ""):
            wass_bary_fast_words = get_wass_bary_fast(params.word1, params.word2, normalized_cluster_hists, word2id,
                                                      ground_metric_matrix, reg=params.regbary)
        else:
            words = params.sent.split(" ")
            wass_bary_fast_words = get_wass_bary_fast_mult(words, normalized_cluster_hists, word2id,
                                                           ground_metric_matrix, mode=params.bary_mode,
                                                           reg=params.regbary)

        if params.nbr_subset == -1:
            closest_words, closest_distances = get_closest_k_wasserstein_words_fast(wass_bary_fast_words, params.top_k,
                                                                                    analysis.id2word_cooc,
                                                                                    normalized_cluster_hists,
                                                                                    ground_metric_matrix, word2id)
        else:
            subset = np.random.choice(analysis.id2word_cooc, params.nbr_subset, replace=False)
            closest_words, closest_distances = get_closest_k_wasserstein_words_fast(wass_bary_fast_words, params.top_k,
                                                                                    subset, normalized_cluster_hists,
                                                                                    ground_metric_matrix, word2id)

        print(closest_words)
        end_time = time.perf_counter()
        logging.info("Computed closest k Wasserstein neighbors!" + " in " + str(end_time - st_time) + " seconds")

    elif params.mode == flags.MODE_SENTENCE_DIST:
        st_time = time.perf_counter()

        sent1 = params.sent1.lower()
        sent2 = params.sent2.lower()
        print("Lowercased sentence 1 ", sent1)
        print("Lowercased sentence 2 ", sent2)

        words1 = sent1.split(" ")
        valid_words1 = []
        for wd in words1:
            if wd in word2id:
                valid_words1.append(wd)

        print("Valid words for sentence 1 are ", [(wd, idx) for idx, wd in enumerate(valid_words1)])
        words2 = sent2.split(" ")
        valid_words2 = []
        for wd in words2:
            if wd in word2id:
                valid_words2.append(wd)
        print("Valid words for sentence 2 are ", [(wd, idx) for idx, wd in enumerate(valid_words2)])

        if params.marginals:
            wts_sent1 = get_wts(marginals_arr, word2id, words1)
            wts_sent2 = get_wts(marginals_arr, word2id, words2)
            _alpha = "smoothed inverse ppmi based weights"
            print("wts_sent1 ", wts_sent1)
            print("wts_sent2 ", wts_sent2)
        else:
            wts_sent1 = None
            wts_sent2 = None
            _alpha = "uniform"

        wass_bary_fast_sent1 = get_wass_bary_fast_mult(words1, normalized_cluster_hists, word2id, ground_metric_matrix,
                                                       mode=params.bary_mode,
                                                       reg=params.regbary, alpha=_alpha, weights=wts_sent1,
                                                       marginals=marginals_arr, smoothing=params.marginal_smoothing)
        wass_bary_fast_sent2 = get_wass_bary_fast_mult(words2, normalized_cluster_hists, word2id, ground_metric_matrix,
                                                       mode=params.bary_mode,
                                                       reg=params.regbary, alpha=_alpha, weights=wts_sent2,
                                                       marginals=marginals_arr, smoothing=params.marginal_smoothing)

        print(get_wass_dist(wass_bary_fast_sent1, wass_bary_fast_sent2, ground_metric_matrix, lambd=params.regdist))

        end_time = time.perf_counter()
        logging.info("Computed wasserstein distance between sentences in " + str(end_time - st_time) + " seconds")


# save to file and print (maybe let's do it later)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Wasserstein distances and barycenters.')
    parser.add_argument('--cooc-root-path', action='store', type=str, required=True,
                        help='path to folder containing cooc pickled files')
    parser.add_argument('--cluster-data-dir', action='store', type=str, required=True,
                        help='path to folder containing cluster data files')
    parser.add_argument('--hists-path', action='store', type=str, required=True,
                        help='path to normalized cluster histogram')
    parser.add_argument('--marginals-path', action='store', type=str,
                        help='path to marginal ppmi values')
    parser.add_argument('--mode', action='store', type=str, required=True, default=flags.MODE_SENTENCE_DIST,
                        help='Mode of operation')
    parser.add_argument('--word1', action='store', type=str, required=False,
                        help='1st Word')
    parser.add_argument('--word2', action='store', type=str, required=False,
                        help='2nd Word')
    parser.add_argument('--regbary', action='store', type=float, default=0.1,
                        help='Regularization strength in Wasserstein barycenter computation')
    parser.add_argument('--regdist', action='store', type=float, default=0.1,
                        help='Regularization strength in Wasserstein distance computation')
    parser.add_argument('--top-k', action='store', type=int, default=200,
                        help='Number of wasserstein neighbors to fetch')
    parser.add_argument('--nbr-subset', action='store', type=int, default=-1,
                        help='Number of wasserstein neighbors to fetch')
    parser.add_argument('--sent', action='store', type=str, default="",
                        help='Sentence: the words of which you compute the barycenter')
    parser.add_argument('--sent1', action='store', type=str, default="",
                        help='1st Sentence')
    parser.add_argument('--sent2', action='store', type=str, default="",
                        help='2nd Sentence')
    parser.add_argument('--marginals', action='store_true',
                        help='If marginals should be used for the weight')
    parser.add_argument('--bary-mode', action='store', type=str, default="plain",
                        help='Mode of Barycenter computation. Possibilities: [plain, btree]')
    parser.add_argument('--marginal-smoothing', action='store', type=float, default=1.0,
                        help='Marginal Smoothing exponent')

    parser.add_argument('--log', action='store', type=str, default="INFO",
                        help='log level')

    args = parser.parse_args()
    main(args)
