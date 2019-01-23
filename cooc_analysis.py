import utils
import struct
import os
import pickle
import time
import flags
import numpy as np
import scipy.sparse as sp


class Analysis:
    def __init__(self, opts):
        self.opts = opts

        self.cooccurr_dic = {}
        self.shuf_cooccurr = []
        self.cooccurrences_df = None
        self.id2word_cooc = []
        self.pickle_ext_str = ".pickle"
        self.cooccurr_str = "cooccurr_dic"
        self.shuf_cooccurr_str = "shuf_cooccurr"
        self.id2word_cooc_str = "id2word_cooc"
        self.load_mode = flags.LOAD_ALL
        self.get_vocab_counts = False

        if 'sparse' in self.opts and self.opts['sparse']:
            self.sparse = True
        else:
            self.sparse = False

        if 'joblib' in self.opts and self.opts['joblib']:
            self.joblib = True
        else:
            self.joblib = False


    def read_cooccurrences(self):
        struct_fmt = 'iid'
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from


        def read_chunks(f, length):
            while True:
                data = f.read(length)
                if not data:
                    break
                yield data


        with open(self.opts['cooc_fpath'], 'rb') as f:
            self.shuf_cooccurr = [struct_unpack(chunk) for chunk in read_chunks(f, struct_len)]


    def set_id2word_cooc(self):
        if not self.get_vocab_counts:
            with open(os.path.join(self.opts['vocab_fpath'])) as f:
                self.id2word_cooc = [l.rstrip('\n').split(' ')[0] for l in f]
        else:
            with open(os.path.join(self.opts['vocab_fpath'])) as f:
                self.id2word_cooc = [(l.rstrip('\n').split(' ')[0], int(l.rstrip('\n').split(' ')[1])) for l in f]


    def _set_cooccurr_dic(self):
        self.cooccurr_dic = {tuple([self.id2word_cooc[w_id - 1] for w_id in word_ids]): cnt
                             for (*word_ids, cnt) in self.shuf_cooccurr}


    def get_sps_cooc_from_shuffled(self, shuf_cooccurr, num_words):
        print("getting sps cooc from shuffled")
        cooc_len = len(shuf_cooccurr)

        print("cooc_len is ", cooc_len)
        print("num_words is ", num_words)

        row = np.empty((cooc_len,))
        col = np.empty((cooc_len,))
        data = np.empty((cooc_len,))

        idx = 0
        for (*word_ids, cnt) in shuf_cooccurr:
            assert len(word_ids) == 2
            # one based indexing
            row[idx] = word_ids[0] - 1
            col[idx] = word_ids[1] - 1
            data[idx] = cnt
            idx += 1
        print("idx at the end is ", str(idx))
        sps_cooc = sp.csr_matrix((data, (row, col)), shape=(num_words, num_words))
        return sps_cooc


    def _set_cooccurr_sparse(self):
        self.cooccurr_dic = self.get_sps_cooc_from_shuffled(self.shuf_cooccurr, len(self.id2word_cooc))


    def get_cooccurrence_btwn(self, wrd1, wrd2):
        assert not self.sparse
        return self.cooccurr_dic.get((wrd1, wrd2))


    def get_cooccurrences(self, word):
        assert not self.sparse
        return [{word_pair: self.cooccurr_dic[word_pair]}
                for word_pair in self.cooccurr_dic
                if word == word_pair[0]]


    def has_keyword_in_filenames(self, root, keyword):
        for file in os.listdir(root):
            if (keyword in file):
                return True
        return False


    def setup_cooccurr_analysis(self, root=None):
        # print("Root is ", root)
        if root is None or (not self.has_keyword_in_filenames(root, self.pickle_ext_str)):
            print("Loading from cooccurrence binary")
            self.read_cooccurrences()
            self.set_id2word_cooc()
            if not self.sparse:
                self._set_cooccurr_dic()
            else:
                self._set_cooccurr_sparse()
        else:
            self.load_from_pkl(root)


    def load_from_pkl(self, root):
        print("Loading from pickled files")
        for file in os.listdir(root):
            if not self.pickle_ext_str in file:
                continue
            else:
                if ((
                        self.load_mode == flags.LOAD_ALL or self.load_mode == flags.LOAD_INTM) and self.cooccurr_str in file):
                    self.cooccurr_dic = utils.load_cooc(root + file, joblib=self.joblib, sparse=self.sparse,
                                                        nick=self.cooccurr_str)

                elif (self.load_mode == flags.LOAD_ALL and self.shuf_cooccurr_str in file):
                    self.shuf_cooccurr = utils.load_pickle(root + file, nick=self.shuf_cooccurr_str)

                elif self.id2word_cooc_str in file:
                    self.id2word_cooc = utils.load_pickle(root + file, nick=self.id2word_cooc_str)
