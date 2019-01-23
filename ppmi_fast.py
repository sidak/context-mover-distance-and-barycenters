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
    python ppmi_fast.py
        --cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ 
        --smooth 
        --smoothing-parameter 0.15 
        --k-shift 1.0
'''


# **Positive Pointwise Mutual Information (PPMI) calculation**
# 
# PPMI(w, c) = max(0, PMI(w, c))
# 
# PMI(w, c) = log(p(w, c)/(p(c)\*p(w))
# 
# p(w, c) = cooc(w, c)/|D|
# 
# p(w) = Sum_over_c(cooc(w,c))/|D|
# 
# |D| = Sum_over_w(Sum_over_c(cooc(w,c)))

def get_sps_cooc(cooccurr_dic, word2id):
    cooc_len = len(cooccurr_dic)
    num_words = len(word2id)

    print("cooc_len is ", cooc_len)
    print("num_words is ", num_words)

    row = np.empty((cooc_len,))
    col = np.empty((cooc_len,))
    data = np.empty((cooc_len,))

    idx = 0
    for word1, word2 in cooccurr_dic:
        if word1 in word2id and word2 in word2id:
            row[idx] = word2id[word1]
            col[idx] = word2id[word2]
            data[idx] = cooccurr_dic[(word1, word2)]
            idx += 1
    print("idx at the end is ", str(idx))
    sps_cooc = sp.csr_matrix((data, (row, col)), shape=(num_words, num_words))
    return sps_cooc


def compute_ppmi_sps(sps_ppmi_arr, smoothing_parameter=0.75, k_shift=1.0):
    _total_word_cooc = sps_ppmi_arr.sum(axis=1)
    _context_word_corrected_cooc = np.power(_total_word_cooc, smoothing_parameter)
    _context_word_corrected_cooc = _context_word_corrected_cooc.reshape((1, -1))
    sps_ppmi_arr = sps_ppmi_arr.multiply(1.0 / _total_word_cooc)
    _context_word_corrected_cooc_sum = _context_word_corrected_cooc.sum()
    sps_ppmi_arr *= _context_word_corrected_cooc_sum
    sps_ppmi_arr = sps_ppmi_arr.multiply(1.0 / _context_word_corrected_cooc)
    sps_ppmi_arr = sps_ppmi_arr._with_data(np.log(sps_ppmi_arr.data), copy=False)
    if (k_shift > 1.5):
        sps_ppmi_arr.data -= np.log(k_shift)
    sps_ppmi_arr.data = np.clip(sps_ppmi_arr.data, a_min=0.0, a_max=None)
    return sps_ppmi_arr


def main(args):
    params = utils.dotdict(vars(args))
    print('-------- PARAMETERS --------')
    print(json.dumps(params, sort_keys=True, indent=4))
    print('----------------------------')

    logging.basicConfig()
    if params.log:
        level = logging.getLevelName(params.log)
        logging.getLogger().setLevel(level)

    logging.info("Load co-occurrence pickled files")

    opts = {}
    if params.sparse_cooc:
        opts['sparse'] = True

    analysis = Analysis(opts=opts)
    if params.remove_pain:
        params.cooc_sps_matrix_dump_name = params.cooc_sps_matrix_dump_name + "_no_pain"

    params.cooc_sps_matrix_dump_name = params.cooc_sps_matrix_dump_name + ".npz"
    print("Name of cooc_sps_matrix_dump_name ", params.cooc_sps_matrix_dump_name)
    sps_matrix_exists = params.cooc_sps_matrix_dump_name in os.listdir(params.cooc_root_path)

    if (sps_matrix_exists):
        logging.info("Yes, cooc_sps_matrix exists")
        st_time = time.perf_counter()
        cooc_sps_matrix = sp.load_npz(os.path.join(params.cooc_root_path, params.cooc_sps_matrix_dump_name))
        analysis.load_mode = flags.LOAD_MIN
        end_time = time.perf_counter()
        logging.info("Loaded cooc_sps_matrix!" + " in " + str(end_time - st_time) + " seconds")
        if args.cooc_marginals_only:
            st_time = time.perf_counter()
            marginals_cooc = np.asarray(cooc_sps_matrix.sum(axis=1))

            if params.save_dir != "":
                dest_dir = os.path.join(params.cooc_root_path, params.save_dir)
            else:
                dest_dir = params.cooc_root_path

            np.savez(os.path.join(dest_dir, 'cooc_marginals.npz'), marginals_cooc)
            end_time = time.perf_counter()
            logging.info("Dumped cooc_sps_matrix marginals!" + " in " + str(end_time - st_time) + " seconds")
            return
    else:
        logging.info("No, cooc_sps_matrix doesn't exists")
        analysis.load_mode = flags.LOAD_INTM

    st_time = time.perf_counter()
    analysis.setup_cooccurr_analysis(params.cooc_root_path)
    word2id = {word: idx for idx, word in enumerate(analysis.id2word_cooc)}
    end_time = time.perf_counter()
    logging.info("Loaded cooc_analysis!" + " in " + str(end_time - st_time) + " seconds")

    if (not sps_matrix_exists):
        st_time = time.perf_counter()

        if params.remove_pain:
            word_vector_dict = pickle.load(open(os.path.join(params.cluster_path, 'word_vector.pickle'), 'rb'))
            painful_words = []
            print("len of word2id before is ", len(word2id))
            for wd in word2id:
                if wd not in word_vector_dict:
                    painful_words.append(wd)
            print("Painful words are ", painful_words)
            for wd in painful_words:
                word2id.pop(wd)
            print("len of word2id after is ", len(word2id))
            sorted_word2d = sorted(list(word2id.items()), key=lambda item: item[1])  # sorted list of tuples
            sorted_word_list = list(zip(*sorted_word2d))[
                0]  # 'zip(*)' unzips & we have a list containing 2 tuples, list[0] = idxs, list[0]=vals
            word2id = dict(zip(sorted_word_list, range(len(sorted_word_list))))
            print("Sorted word2id now, rechecking len of it ", len(sorted_word_list))

        if params.sparse_cooc:
            cooc_sps_matrix = analysis.cooccurr_dic
        else:
            cooc_sps_matrix = get_sps_cooc(analysis.cooccurr_dic, word2id)

        mid_time = time.perf_counter()
        logging.info("Created cooc_sps_matrix!" + " in " + str(mid_time - st_time) + " seconds")
        sp.save_npz(os.path.join(params.cooc_root_path, params.cooc_sps_matrix_dump_name), cooc_sps_matrix)
        end_time = time.perf_counter()
        logging.info(
            "Dumped cooc_sps_matrix for fast loading in future!" + " in " + str(end_time - mid_time) + " seconds")

    if not params.multiple:
        smoothing_list = [params.smoothing_parameter]
    else:
        smoothing_list = params.mult_smoothing_list

    for smoothin in smoothing_list:

        st_time = time.perf_counter()
        ppmi_sps_matrix = compute_ppmi_sps(cooc_sps_matrix, smoothing_parameter=smoothin, k_shift=params.k_shift)
        end_time = time.perf_counter()
        logging.info(
            "Computed ppmi_sps_matrix! for smoothing " + str(smoothin) + " in " + str(end_time - st_time) + " seconds")

        marginals_desc = ""
        pkl_desc = "ppmi"
        if (params.smooth):
            pkl_desc += "_" + "smooth_" + str(smoothin)

        pkl_desc += "_" + "k-shift_" + str(params.k_shift)
        if params.nick != "":
            pkl_desc += "_" + params.nick
        marginals_desc = pkl_desc + "_marginals.npz"
        pkl_desc += ".npz"

        st_time = time.perf_counter()
        marginals_ppmi = np.asarray(ppmi_sps_matrix.sum(axis=1))

        if params.save_dir != "":
            dest_dir = os.path.join(params.cooc_root_path, params.save_dir)
        else:
            dest_dir = params.cooc_root_path

        np.savez(os.path.join(dest_dir, marginals_desc), marginals_ppmi)
        end_time = time.perf_counter()
        logging.info("Dump dense marginals of PPMI for smoothing " + str(smoothin) + " in " + str(
            end_time - st_time) + " seconds")

        if not params.marginals_only:
            logging.info("Also dumping the complete ppmi_sps_matrix, for smoothing " + str(smoothin))
            st_time = time.perf_counter()

            sp.save_npz(os.path.join(dest_dir, pkl_desc), ppmi_sps_matrix)
            end_time = time.perf_counter()
            logging.info(
                "Dumped " + pkl_desc + " for smoothing " + str(smoothin) + " in time: " + str(end_time - st_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get PPMI co-occurrence.')
    parser.add_argument('--cooc-root-path', action='store', type=str, required=True,
                        help='path to folder containing cooc pickled files')
    parser.add_argument('--save-dir', action='store', type=str, default="",
                        help='subdir of cooc-root-path to save files in')
    parser.add_argument('--multiple', action='store_true',
                        help='Compute multiple ppmi')
    parser.add_argument('--mult-smoothing-list', nargs='+', action='store', type=float,
                        help='list of smoothing values to consider')
    parser.add_argument('--smooth', action='store_true',
                        help='If smoothing should be considered')
    parser.add_argument('--smoothing-parameter', action='store', type=float, default=0.75, required=False,
                        help='value of smoothing_parameter alpha ([0, 1])')
    parser.add_argument('--k-shift', action='store', type=float, default=1.0, required=True,
                        help='shifts PPMI score by log(k) with this k')
    parser.add_argument('--nick', action='store', type=str, required=False, default="",
                        help='nick for saving the ppmi')
    parser.add_argument('--marginals-only', action='store_true',
                        help='If only marginals of ppmi matrix should be stored')
    parser.add_argument('--cooc-marginals-only', action='store_true',
                        help='If only marginals of cooc matrix should be stored')
    parser.add_argument('--log', action='store', type=str, default="INFO",
                        help='log level')
    parser.add_argument('--remove-pain', action='store_true',
                        help='Remove painful words!')
    parser.add_argument('--cluster-path', action='store', type=str, required=False,
                        help='for checking getting word_vector_dict path to folder containing cluster pickled files')
    parser.add_argument('--cooc-sps-matrix-dump-name', action='store', type=str, default="cooc_sps_matrix",
                        required=False,
                        help='nick for cooc')
    parser.add_argument('--sparse-cooc', action='store_true',
                        help='Load sparse cooc matrix in cooc_analysis')
    args = parser.parse_args()
    main(args)
