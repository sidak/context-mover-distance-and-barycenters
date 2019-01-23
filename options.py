import argparse


def get_parser(objective='sentence_similarity'):
    parser = argparse.ArgumentParser(description="Context Mover's Distance & Barycenters")
    add_core_args(parser)

    if objective == 'sentence_similarity':
        add_sentence_similarity_args(parser)
    else:
        raise NotImplementedError

    add_barycenter_args(parser)
    add_sinkhorn_args(parser)
    add_marginal_args(parser)
    add_groundmetric_args(parser)
    add_filterwords_args(parser)

    return parser


def add_core_args(parser):
    group = parser.add_argument_group('Core args needed for setting up all tasks')

    group.add_argument('--cooc-root-path', action='store', type=str, required=True,
                       help='path to folder containing cooc pickled files')
    group.add_argument('--cluster-data-dir', action='store', type=str, required=True,
                       help='path to folder containing cluster data files')
    group.add_argument('--hists-path', action='store', type=str, required=True,
                       help='path to normalized cluster histogram')
    group.add_argument('--debug', action='store_true',
                       help='Launch in debugging mode')
    return group


def add_marginal_args(parser):
    group = parser.add_argument_group('Related to marginals used for computing weights for barycenter')

    group.add_argument('--marginals', action='store_true',
                       help='If marginals should be used for the weight')
    group.add_argument('--marginals-path', action='store', type=str,
                       help='path to marginal ppmi values')
    group.add_argument('--marginals-file', action='store', type=str,
                       help='path to freq file (needed for sif type marginals)')
    group.add_argument('--marginal-smoothing', action='store', type=float, default=1.0,
                       help='Marginal Smoothing exponent')
    group.add_argument('--marginal-a', action='store', type=float, default=1e-3,
                       help='Marginal A')
    group.add_argument('--marginal-type', action='store', type=str, default='ppmi',
                       help='Marginal kind: ppmi, sif')
    group.add_argument('--marginal-log', action='store_true',
                       help='Do log of marginals')
    return group


def add_barycenter_args(parser):
    group = parser.add_argument_group('Barycenter specific')

    group.add_argument('--regbary', action='store', type=float, default=0.1,
                       help='Regularization strength in Wasserstein barycenter computation')
    group.add_argument('--bary-mode', action='store', type=str, default="plain",
                       help='Mode of Barycenter computation. Possibilities: [plain, btree]')
    group.add_argument('--seq', action='store_true',
                       help='Compute barycenter sequentially')
    group.add_argument('--length_correction', action='store_true',
                       help='Do length correction')
    group.add_argument('--weighting', action='store', type=str, default="uniform",
                       help='Weighting to use for each histogram in barycenter computation. \
						Possibilities: ["cooc_inv", "ppmi_direct", "uniform", "hm_cooc_btree"]')
    return group


def add_sinkhorn_args(parser):
    group = parser.add_argument_group('Sinkhorn specific')

    group.add_argument('--thresh', action='store', type=float, default=1e-09,
                       help='Threshold for sinkhorn iterations')
    group.add_argument('--max-its', action='store', type=int, default=100,
                       help='Max Number of sinkhorn iterations to perform (default:100)')
    return group


def add_sentence_similarity_args(parser):
    group = parser.add_argument_group('Sentence similarity specific args')

    group.add_argument('--tasks', action='store', type=str, required=True, default="STS15",
                       help='Tasks to run on')
    group.add_argument('--save-result-file', action='store', type=str, required=True,
                       help='path to file containing the saved results')
    group.add_argument('--gpu-id', action='store', default=-1, type=int,
                       help='GPU core to use (default: -1 [disabled])')
    group.add_argument('--sts-batch-size', action='store', default=32, type=int,
                       help='batch size during sts evaluation (useful to control in gpu mode) (default: 32)')
    group.add_argument('--similarity', action='store', type=str, default="wasserstein",
                       help='Similarity/Distance to use between histograms. Possibilities: [wasserstein, wass_cosine]')
    group.add_argument('--normalize', action='store_true',
                       help='If cluster centers vectors should be normalized')
    group.add_argument('--similarity-batched', action='store_true',
                       help='Compute wass dist in batched mode ')
    group.add_argument('--barycenter-batched', action='store_true',
                       help='Compute wass barycenters in batched mode ')
    group.add_argument('--double', action='store_true',
                       help='Use double for barycenter computations')
    group.add_argument('--wasserstein-kp', action='store_true',
                       help='Use kernel product based method for calculating wasserstein distance')
    group.add_argument('--save-compute-kp', action='store_true',
                       help='Save compute of ground matrix in kp')
    group.add_argument('--interpolate-repr', action='store', type=float, default=-1,
                       help='Interpolate between point and distribution representations, where \
						the value indicates how much contribution of point representations to use ')
    group.add_argument('--interpolate-complete', action='store_true',
                       help='Interpolate starting from the calculation of barycenters')
    group.add_argument('--interpolate-reduced', action='store_true',
                       help='Interpolate with reduced (weighted avg) point embeddings \
						and just use it in distance calculations (and not barycenter)')
    group.add_argument('--barycentric-interpolate-reduced', action='store_true',
                       help='Interpolate in barycentric manner with reduced (weighted avg) point embeddings \
						and just use it in distance calculations (and not barycenter)')
    group.add_argument('--interpolate-wts-uniform', action='store_true',
                       help='Uniform wts for point embeddings while interpolate (intended for sent2vec')
    group.add_argument('--interpolate-reduced-pc', action='store', type=int, default=-1,
                       help='Number of principal components to remove in interpolate-reduced (-1 means not done)')

    return group


def add_groundmetric_args(parser):
    group = parser.add_argument_group('Ground Metric specific')

    group.add_argument('--gm-normalize', action='store', type=str, default=None,
                       help='Normalize ground metric. Possibilities: [max, min, median]')
    group.add_argument('--gm-type', action='store', type=str, default="euclidean",
                       help='ground metric type to use. Possibilities: [euclidean, cosine, angular]')
    group.add_argument('--clip-gm', action='store_true', help='to clip ground metric')
    group.add_argument('--clip-min', action='store', type=float, default=0,
                       help='Value for clip-min for gm')
    group.add_argument('--clip-max', action='store', type=float, default=5,
                       help='Value for clip-max for gm')
    group.add_argument('--wass2', action='store_true',
                       help='Compute wasserstein-2 i.e. C(x, y) = ||x-y||^2. Default: Wasserstein-1')
    return group


def add_filterwords_args(parser):
    group = parser.add_argument_group('Filtering words while computing barycenter')

    parser.add_argument('--filter-freq-relative', action='store_true',
                        help='Filter words by freq relative to the sentence in barycenter')
    parser.add_argument('--filter-mult', action='store', type=float, default=1.0,
                        help='This is the x in 1/(L.x) which is used as a threshold to filter.')
    parser.add_argument('--filter-freq', action='store_true',
                        help='Filter words by freq in barycenter')
    parser.add_argument('--filter-thresh', action='store', type=float, default=1e-05,
                        help='Threshold for taking word into barycenter computation')
    return group
