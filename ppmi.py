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

'''
Usage
    python ppmi.py 
        --cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ 
        --smooth 
        --smoothing-parameter 0.75 
'''


def get_ppmi(cooccurr_dic, word_cooc, context_word_cooc, total_cooc, word1, word2):
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

    numer = cooccurr_dic[(word1, word2)] * total_cooc
    denom = word_cooc[word1] * context_word_cooc[word2]
    return max(0, math.log2(numer / denom))


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

    analysis = Analysis(opts=opts)
    analysis.load_all = False
    analysis.setup_cooccurr_analysis(params.cooc_root_path)

    logging.info("Done loading pickled!")
    st_time = time.perf_counter()
    total_word_cooc = {word: 0.0 for word in analysis.id2word_cooc}

    for word1, word2 in analysis.cooccurr_dic:
        total_word_cooc[word1] += analysis.cooccurr_dic[(word1, word2)]

    total_cooc = np.sum(list(total_word_cooc.values()))

    end_time = time.perf_counter()
    logging.info("Created total_cooc!" + " in " + str(end_time - st_time) + " seconds")

    if (not params.smooth):
        ppmi_dic = {(word1, word2): get_ppmi(analysis.cooccurr_dic, total_word_cooc, total_word_cooc, total_cooc,
                                             word1, word2) for word1, word2 in analysis.cooccurr_dic}
        logging.info("Created ppmi_dic")
    else:
        st_time = time.perf_counter()
        context_word_corrected_cooc = {word: total_word_cooc[word] ** params.smoothing_parameter for word in
                                       total_word_cooc}
        context_word_corrected_cooc_sum = np.sum(list(context_word_corrected_cooc.values()))
        end_time = time.perf_counter()
        logging.info("Created context_word_corrected_cooc!" + " in " + str(end_time - st_time) + " seconds")

        st_time = time.perf_counter()
        ppmi_dic = {(word1, word2): get_ppmi(analysis.cooccurr_dic, total_word_cooc, context_word_corrected_cooc,
                                             context_word_corrected_cooc_sum,
                                             word1, word2) for word1, word2 in analysis.cooccurr_dic}
        end_time = time.perf_counter()
        logging.info("Created smoothed_ppmi_dic" + " in " + str(end_time - st_time) + " seconds")

    pkl_desc = "ppmi"
    if (params.smooth):
        pkl_desc += "_" + "smooth_" + str(params.smoothing_parameter)

    if (params.filter_zero):
        logging.info("len of ppmi before " + str(len(ppmi_dic)))
        ppmi_dic = dict(filter(lambda x: x[1] > 0, ppmi_dic.items()))
        logging.info("len of ppmi after filtering" + str(len(ppmi_dic)))
        pkl_desc += "_filter_zero"

    pkl_time = utils.pickle_obj(ppmi_dic, os.path.join(params.cooc_root_path, pkl_desc + ".pickle"))
    logging.info("Pickled " + pkl_desc + " in time: " + str(pkl_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get PPMI co-occurrence.')
    parser.add_argument('--cooc-root-path', action='store', type=str, required=True,
                        help='path to folder containing cooc pickled files')
    parser.add_argument('--smooth', action='store_true',
                        help='If smoothing should be considered')
    parser.add_argument('--filter-zero', action='store_true',
                        help='Filter out elements from dic that are zero')
    parser.add_argument('--smoothing-parameter', action='store', type=float, default=0.75, required=True,
                        help='value of smoothing_parameter alpha ([0, 1])')

    parser.add_argument('--log', action='store', type=str, default="INFO",
                        help='log level')
    args = parser.parse_args()
    main(args)
