# Context Mover’s Distance & Barycenters: Optimal transport of contexts for building representations

## Abstract
We present a framework for building unsupervised representations of entities and their compositions, where each entity is viewed as a probability distribution rather than a fixed length vector. In particular, this distribution is supported over the contexts which co-occur with the entity and are embedded in a suitable low-dimensional space. This enables us to consider the problem of representation learning with a perspective from Optimal Transport and take advantage of its numerous tools such as Wasserstein distance and Wasserstein barycenters. We elaborate how the method can be applied for obtaining unsupervised representations of text and illustrate the performance quantitatively as well as qualitatively on tasks such as measuring sentence similarity and word entailment, where we empirically observe significant gains. 

The key benefits of the proposed approach include: \
(a) capturing uncertainty and polysemy via modeling the entities as distributions \
(b) utilizing the underlying geometry of the particular task (with the ground cost) \
(c) simultaneously providing interpretability with the notion of optimal transport between contexts and \
(d) easy applicability on the top of existing point embedding methods. In essence, the framework can be useful for any unsupervised or supervised problem (on text or other modalities); and only requires a co-occurrence structure inherent to many problems. 

## Pre-trained co-occurrences/histograms/vectors

The co-occurrence information used for both sentence and entailment experiments can be found in the following directory on Google Drive. The sentence experiments were carried out on Toronto Book Corpus, and the entailment ones on a Wikipedia dump. 

https://drive.google.com/open?id=13stRuUd--71hcOq92yWUF-0iY15DYKNf

In particular, one can find the sparse PPMI matrices, the pre-trained vectors from GloVe and Henderson, as well as the clusters computed with kmcuda. Based on which any of the experiments can be reproduced with the available code. Further, we provide some pre-computed histograms, making it easy for the users to directly utilize the distributional estimates. 



## Usage
#### clustering.py
```bash
python clustering.py    --input-vector-file ./data/vectors/entailment_vectors_200.glove \
                        --algo kmeans \
                        --clusters-to-make 100 \ 
                        --vocab-file ./data/vectors/entailment_vocab_200.glove \ 
                        --target-folder ./data/vectors/clusters/entailment_vectors_200 \
                        --dim 200
```

#### ppmi_fast.py
```bash
python ppmi_fast.py     --cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/  \
                        --smooth \
                        --smoothing-parameter 0.15 \ 
                        --k-shift 1.0
```


#### ppmi.py
```bash
python ppmi.py  --cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ \ 
                --smooth \
                --smoothing-parameter 0.75 

```


#### histograms.py
```bash
python histograms.py    --cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/  \
                        --ppmi-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ppmi_smooth_0.75_k-shift_1.0.npz \
                        --cluster-data-dir ./data/vectors/clusters/book_glove_symmetric=1_window-size=10_min-count=10_eta=0.005_iter=75_cleaned=300/kmeans_100_205513 \
                        --histogram-data-dir ./data/histograms/
```

#### cooc_pickler.py
```bash
python cooc_pickler.py  --cooc-fpath ./data/cooc/book_coocurrence_symmetric=1_window-size=10_cleaned=300.bin \ 
                        --vocab-fpath ./data/corpus/book_vocab_min-count=10_cleaned=300.txt
```


#### wasserstein.py
```bash
python wasserstein.py   --cluster-data-dir ./data/vectors/clusters/book_glove_symmetric=1_window-size=10_min-count=10_eta=0.005_iter=75_cleaned=300/kmeans_100_205513 \
                        --cooc-root-path ./data/cooc/book_coocurrence_symmetric\=1_window-size\=10_cleaned\=300/ \
                        --hists-path ./data/histograms/ppmi_smooth_0.75_k-shift_1.0_kmeans_100_205513/normalized_cluster_hists.npz \
                        --word1 cute --word2 animal \
                        --marginals-path ./data/cooc/book_coocurrence_symmetric=1_window-size=10_cleaned=300/ppmi_smooth_0.75_k-shift_1.0_marginals.npz

```