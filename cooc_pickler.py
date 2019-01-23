import pickle
import os
from cooc_analysis import Analysis
import argparse
import utils
import json
import logging

'''
Usage
python cooc_pickler.py 
    --cooc-fpath "./data/cooc/book_coocurrence_symmetric=1_window-size=10_cleaned=300.bin" 
    --vocab-fpath "./data/corpus/book_vocab_min-count=10_cleaned=300.txt"
'''


def main(args):
    params = utils.dotdict(vars(args))
    print('-------- PARAMETERS --------')
    print(json.dumps(params, sort_keys=True, indent=4))
    print('----------------------------')

    logging.basicConfig()
    if params.log:
        level = logging.getLevelName(params.log)
        logging.getLogger().setLevel(level)

    logging.info("Setup_cooccurr_analysis")

    opts = {}
    opts['cooc_fpath'] = params.cooc_fpath
    opts['vocab_fpath'] = params.vocab_fpath
    opts['joblib'] = params.joblib
    opts['sparse'] = params.sparse
    analysis = Analysis(opts=opts)
    analysis.setup_cooccurr_analysis()

    logging.info("Done with cooccur reading and analysis. Starting pickling!")

    cooc_bin_name = params.cooc_fpath.split('/')[-1]
    cooc_folder_name = params.cooc_fpath.replace(cooc_bin_name, '')
    cooc_bin_name_without_ext = cooc_bin_name.replace('.bin', '')

    if params.joblib:
        save_cooc = utils.joblib_obj
    elif params.sparse:
        save_cooc = utils.sparse_obj
    else:
        save_cooc = utils.pickle_obj

    pkl_time = save_cooc(analysis.cooccurr_dic, os.path.join(cooc_folder_name,
                                                             analysis.cooccurr_str + "_" + cooc_bin_name_without_ext + ".pickle"))
    logging.info("Pickled cooccurr_dic in time: " + str(pkl_time))

    if not params.skip_shuf_cooc:
        pkl_time = utils.pickle_obj(analysis.shuf_cooccurr, os.path.join(cooc_folder_name,
                                                                         analysis.shuf_cooccurr_str + "_" + cooc_bin_name_without_ext + ".pickle"))
        logging.info("Pickled shuf_cooccurr in time: " + str(pkl_time))

    pkl_time = utils.pickle_obj(analysis.id2word_cooc, os.path.join(cooc_folder_name,
                                                                    analysis.id2word_cooc_str + "_" + cooc_bin_name_without_ext + ".pickle"))
    logging.info("Pickled id2word_cooc in time: " + str(pkl_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load cooccurrence binary and pickle.')
    parser.add_argument('--cooc-fpath', action='store', type=str, required=True,
                        help='path to co-occurrence binary')
    parser.add_argument('--vocab-fpath', action='store', type=str, required=True,
                        help='path to vocabulary')
    parser.add_argument('--log', action='store', type=str, default="INFO",
                        help='log level')
    parser.add_argument('--joblib', action='store_true', default=False,
                        help='use joblib to dump objects')
    parser.add_argument('--sparse', action='store_true', default=False,
                        help='use scipy sparse to read and dump the cooc-matrix!')
    parser.add_argument('--skip-shuf-cooc', action='store_true', default=False,
                        help='skip pickling shuf_cooc (for large datasets)')
    args = parser.parse_args()
    main(args)
