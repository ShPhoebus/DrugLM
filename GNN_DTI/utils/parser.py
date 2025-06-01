import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MixGCF")

    parser.add_argument("--dataset", nargs="?", default="drug",
                        help="Choose a dataset:[drug]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument(
        "--pretrain_path", nargs="?", default="embeddings/train_embeddings.pt",
        help="Path to pretrained embeddings"
    )
    parser.add_argument(
        "--use_pretrain", type=bool, default=False,
        help="Whether to use pretrained embeddings"
    )
    parser.add_argument(
        "--embedding-file", type=str, required=True,
        help="Path to LM embedding file (.pt). This parameter is required."
    )

    parser.add_argument("--gnn", nargs="?", default="lightgcn",
                        help="Choose a recommender:[lightgcn, ngcf]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=1024, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='concat', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
#     parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 20, 40, 60]',
#                         help='Output sizes of every layer')
    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 20, 40, 60]',
                        help='Output sizes of every layer')
#     parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
#                         help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=3, help="hop")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./weights/", help="output directory for model"
    )
    
    parser.add_argument(
        "--save_emb_epochs", type=str, default="", 
        help="Specify epochs to save embeddings, separated by commas, e.g. '10,20,30'"
    )

    return parser.parse_args()
