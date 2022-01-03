import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader

from sklearn.metrics import average_precision_score as average_precision

import sys
import argparse
import h5py
import subprocess as sp
import numpy as np
import pandas as pd
import gzip as gz

import src.fasta as fa
from tqdm import tqdm
from src.alphabets import Uniprot21
from src.utils import PairedDataset, collate_paired_sequences
from src.models.embedding import LastHundredEmbed, IdentityEmbed, FullyConnectedEmbed
from src.models.contact import ContactCNN
from src.models.interaction import ModelInteraction
from src.models.glider import glide_compute_map

def predict_interaction(model, n0, n1, tensors, use_cuda):

    b = len(n0)

    p_hat = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
#         z_a = torch.Tensor(z0[i][:,-embedding_size:])
#         z_b = torch.Tensor(z1[i][:,-embedding_size:])
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()

#         z_a = z_a.unsqueeze(0)
#         z_b = z_b.unsqueeze(0)
        p_hat.append(model.predict(z_a, z_b))
    p_hat = torch.stack(p_hat, 0)
    return p_hat

def predict_cmap_interaction(model, n0, n1, tensors, use_cuda):

    b = len(n0)

    p_hat = []
    c_map_mag = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()

#         z_a = z_a.unsqueeze(0)
#         z_b = z_b.unsqueeze(0)
        cm, ph = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
    p_hat = torch.stack(p_hat, 0)
    c_map_mag = torch.stack(c_map_mag, 0)
    return c_map_mag, p_hat

def interaction_grad(model, n0, n1, y, tensors, use_cuda, glider_map = None, glider_mat = None, weight=0.35, skew=0.5, alpha=0):

    def glider_score(p, q):
        if glider_map is None or glider_mat is None:
            return model.glider_score(p, q)
        for m in [p, q]:
            if m not in glider_map:
                return 0
            p_ = glider_map[p]
            q_ = glider_map[q]
            return glider_mat[p_, q_]
    
    c_map_mag, p_hat = predict_cmap_interaction(model, n0, n1, tensors, use_cuda)
    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    if model.use_glider:
        n_edges = len(n0)
        g_score = []
        p_res   = []
        for i in range(n_edges):
            g_score.append(torch.tensor(glider_score(n0[i], n1[i]), dtype = torch.float64))
        g_score = torch.stack(g_score, 0)
        if use_cuda:
            g_score = g_score.cuda()

    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
    combined_loss = bce_loss

    if model.use_glider:
        glider_loss = F.binary_cross_entropy(p_hat.float(), g_score.float())
        combined_loss = model.glider_param * glider_loss + (1 - model.glider_param) * bce_loss

    cmap_loss = torch.mean(c_map_mag)
    loss = (weight * combined_loss) + ((1-weight) * cmap_loss)
    b = len(p_hat)

    # backprop loss
    loss.backward()

    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()

    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        y = y.float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat)**2).item()

    return loss, correct, mse, b

def interaction_eval(model, test_iterator, tensors, use_cuda):
    p_hat = []
    true_y = []

    for n0, n1, y in test_iterator:
        p_hat.append(predict_interaction(model, n0, n1, tensors, use_cuda))
        true_y.append(y)

    y = torch.cat(true_y, 0)
    p_hat = torch.cat(p_hat, 0)

    if use_cuda:
        y.cuda()
        p_hat = torch.Tensor([x.cuda() for x in p_hat])
        p_hat.cuda()

    loss = F.binary_cross_entropy(p_hat.float(), y.float()).item()
    b = len(y)

    with torch.no_grad():
        guess_cutoff = torch.Tensor([0.5]).float()
        p_hat = p_hat.float()
        y = y.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat)**2).item()

        tp = torch.sum(y*p_hat).item()
        pr = tp/torch.sum(p_hat).item()
        re = tp/torch.sum(y).item()
        f1 = 2*pr*re/(pr + re)

    y = y.cpu().numpy()
    p_hat = p_hat.data.cpu().numpy()

    aupr = average_precision(y, p_hat)

    return loss, correct, mse, pr, re, f1, aupr

def get_embeddings(f, n0, n1,thresh=800):
    z0 = []
    z1 = []

    for (n_a, n_b) in zip(n0, n1):
        z_a = f[n_a]
        z_b = f[n_b]
        if len(z_a) > thresh or len(z_b) > thresh:
            continue
        z0.append(z_a)
        z1.append(z_b)
    assert len(z0)>0 and len(z1)>0 and len(z0) == len(z1), (len(z0), len(z1))
    return z0, z1

def get_pair_names(path):
    n0, n1 = [], []
    with open(path,'r') as f:
        for l in f:
            a,b = l.strip().split()[:2]
            n0.append(a)
            n1.append(b)
    return n0, n1

def parse_args():
    # Defaults
    DATASET_TO_USE = 'tiny'

    ptr_default = '/scratch1/samsl/STRING/homo.sapiens/{}/OLD_DATA.train.pos'.format(DATASET_TO_USE, DATASET_TO_USE)
    ntr_default = '/scratch1/samsl/STRING/homo.sapiens/{}/OLD_DATA.train.neg'.format(DATASET_TO_USE, DATASET_TO_USE)
    pte_default = '/scratch1/samsl/STRING/homo.sapiens/{}/OLD_DATA.dev.pos'.format(DATASET_TO_USE, DATASET_TO_USE)
    nte_default = '/scratch1/samsl/STRING/homo.sapiens/{}/OLD_DATA.dev.neg'.format(DATASET_TO_USE, DATASET_TO_USE)
    emb_default = '/scratch1/tbepler/data/STITCH/homo.sapiens.seqs.ssa_L1_100d_skip_lstm3x1024_uniref90_iter01000000_dlm_sim_tau0.5_augment0.05_mb64_contacts_both_mb16_0.1_0.9_0.5_iter1000000.h5'

    parser = argparse.ArgumentParser('Script for training protein interaction prediction model')
    #parser.add_argument('--config', help='yaml config file - command line options supercede config file')

    # Data
    parser.add_argument('--pos-pairs', default = ptr_default, help='list of true positive pairs')
    parser.add_argument('--neg-pairs', default = ntr_default, help='list of true negative (or random) pairs')
    parser.add_argument('--pos-test',  default = pte_default, help='list of true positive pairs for evaluation')
    parser.add_argument('--neg-test',  default = nte_default, help='list of true negative (or random) pairs for evaluation')
    parser.add_argument('--embedding', default=emb_default, help='h5py path containing embedded sequences')
    parser.add_argument('--augment',action='store_true',help='set flag to augment data by adding (B A) for all pairs (A B)')
    #parser.add_argument('--embedding-size', default=100, help='dimension of embeddings (default: 100)')
    parser.add_argument('--protein-size', default=800, help='maximum protein size to use in training data (default: 800)')

    # Embedding model
    parser.add_argument('--dumb-embed-switch',action='store_true', help='use last 100 dimensions of embedding naively')
    parser.add_argument('--projection-dim', type=int, default=100, help='dimension of embedding projection layer (default: 100)')
    parser.add_argument('--dropout-p', type=float, default=0.2, help='parameter p for embedding dropout layer (default: 0.2)')

    # Contact model
    parser.add_argument('--hidden-dim', type=int, default=50, help='number of hidden units for comparison layer in contact prediction (default: 50)')
    parser.add_argument('--kernel-width', type=int, default=7, help='width of convolutional filter for contact prediction (default: 7)')

    # Interaction Model
    parser.add_argument('--use-w', action='store_true', help='use weight matrix in interaction prediction model')
    parser.add_argument('--do-pool', action='store_true', help='use max pool layer in interaction prediction model')
    parser.add_argument('--pool-width', type=int, default=9, help='size of max-pool in interaction model (default: 9)')
    parser.add_argument('--sigmoid', action='store_true', help='use sigmoid activation at end of interaction model')

    # Training
    parser.add_argument('--negative-ratio', type=int, default=10, help='number of negative training samples for each positive training sample (default: 10)')
    parser.add_argument('--epoch-scale', type=int, default=5, help='report heldout performance every this many epochs (default: 5)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs (default: 100)')

    parser.add_argument('--batch-size', type=int, default=25, help='minibatch size (default: 25)')
    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.05, help='weight on the similarity objective (default: 0.65)')
    parser.add_argument('--pred-skew',type=float,default=0.5,help='raise predictions to pred_skew before computing loss (default: 0.5)')
    parser.add_argument('--skew-alpha',type=float,default=0,help='p = (alpha)(phat) + (1-alpha)(phat ** skew) (default: 0)')

    # GLIDE
    parser.add_argument(
        "--use_glider",
        action = "store_true",
        default= False,
        help="Use GLIDER or Not",
    )

    parser.add_argument(
        "--glider_param",
        type = float,
        default= 0.2,
        help="Glider Training Param",
    )


    parser.add_argument(
        "--glider_thresh",
        type = float,
        default= 92.5,
        help="Glider Cutoff Threshold",
    )


    # Output
    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-1, help='compute device to use')
    parser.add_argument('--checkpoint', help='checkpoint model to start training from')

    args = parser.parse_args()
    return args

def load_embeddings_from_args(args, output):
    ## Create data sets
    batch_size = args.batch_size

    pos_train = args.pos_pairs
    neg_train = args.neg_pairs
    pos_test = args.pos_test
    neg_test = args.neg_test
    augment = args.augment
    protein_size = args.protein_size

    embedding_h5 = args.embedding
    h5fi = h5py.File(embedding_h5, 'r')

    print(f'# Loading training pairs from {pos_train},{neg_train}...', file=output)
    output.flush()

    n0_pos, n1_pos = get_pair_names(pos_train)
    #z0_pos, z1_pos = get_embeddings(h5fi, n0_pos, n1_pos,thresh=protein_size)
    if augment:
        n0_pos_new = n0_pos + n1_pos
        n1_pos_new = n1_pos + n0_pos
        n0_pos = n0_pos_new
        n1_pos = n1_pos_new
    y_pos = torch.ones(len(n0_pos), dtype=torch.float32)

    print(f'# Loaded {len(n0_pos)} positive training pairs', file=output)
    output.flush()

    n0_neg, n1_neg = get_pair_names(neg_train)
    #z0_neg, z1_neg = get_embeddings(h5fi, n0_neg, n1_neg,thresh=protein_size)
    if augment:
        n0_neg_new = n0_neg + n1_neg
        n1_neg_new = n1_neg + n0_neg
        n0_neg = n0_neg_new
        n1_neg = n1_neg_new
    y_neg = torch.zeros(len(n0_neg), dtype=torch.float32)

    print(f'# Loaded {len(n0_neg)} negative training pairs', file=output)

    print(f'# Loading testing pairs from {pos_test},{neg_test}...', file=output)
    output.flush()

    n0_pos_test, n1_pos_test = get_pair_names(pos_test)
    #z0_pos_test, z1_pos_test = get_embeddings(h5fi, n0_pos_test, n1_pos_test,thresh=protein_size)
    y_pos_test = torch.ones(len(n0_pos_test), dtype=torch.float32)

    print(f'# Loaded {len(n0_pos_test)} positive test pairs', file=output)
    output.flush()

    n0_neg_test, n1_neg_test = get_pair_names(neg_test)
    #z0_neg_test, z1_neg_test = get_embeddings(h5fi, n0_neg_test, n1_neg_test,thresh=protein_size)
    y_neg_test = torch.zeros(len(n0_neg_test), dtype=torch.float32)

    print(f'# Loaded {len(n0_neg_test)} negative test pairs', file=output)
    output.flush()

    train_pairs = PairedDataset(n0_pos+n0_neg, n1_pos+n1_neg, torch.cat((y_pos, y_neg), 0))
    pairs_train_iterator = torch.utils.data.DataLoader(train_pairs,
                                                        batch_size=batch_size,
                                                        collate_fn=collate_paired_sequences,
                                                        shuffle=True)

    test_pairs = PairedDataset(n0_pos_test+n0_neg_test, n1_pos_test+n1_neg_test, torch.cat((y_pos_test, y_neg_test), 0))
    pairs_test_iterator = torch.utils.data.DataLoader(test_pairs,
                                                       batch_size=batch_size,
                                                       collate_fn=collate_paired_sequences,
                                                       shuffle=True)

    output.flush()

    print(f'# Loading Tensors',file=output)
    tensors = {}
    all_proteins = set(n0_pos).union(set(n1_pos)).union(set(n0_neg)).union(set(n1_neg)).union(set(n0_pos_test)).union(set(n1_pos_test)).union(set(n0_neg_test)).union(set(n1_neg_test))
    for prot_name in tqdm(all_proteins):
        if args.dumb_embed_switch:
            tensors[prot_name] = torch.from_numpy(h5fi[prot_name][:,-100:])
        else:
            tensors[prot_name] = torch.from_numpy(h5fi[prot_name][:,:])

    # GLIDE args
    use_glider = args.use_glider
    glider_param = args.glider_param
    glider_thresh = args.glider_thresh
    glider_mat, glider_map = None, None

    pos_pairs_for_glide = list(zip(n0_pos, n1_pos, [1]*len(n0_pos)))
    if use_glider:
        print('# Computing GLIDER matrix')
        glider_mat, glider_map  = glide_compute_map(pos_pairs_for_glide, thres_p = glider_thresh)

    return pairs_train_iterator, pairs_test_iterator, tensors, glider_mat, glider_map

def run_training_from_args(args, pairs_train_iterator, pairs_test_iterator, tensors, glider_mat, glider_map, output):

    use_cuda = (args.device > -1) and torch.cuda.is_available()

    if args.checkpoint is None:

        # Create embedding model
        if args.dumb_embed_switch:
            print('# Using last 100 dim of embeddings',file=output)
            embedding = None
            projection_dim = 100
        else:
            projection_dim = args.projection_dim
            dropout_p = args.dropout_p
            embedding = FullyConnectedEmbed(6165, projection_dim, dropout = dropout_p)
            print('# Initializing embedding model with:', file=output)
            print(f'\tprojection_dim: {projection_dim}', file=output)
            print(f'\tdropout_p: {dropout_p}', file=output)

        # Create contact model
        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        #embedding_size = args.embedding_size
        print('# Initializing contact model with:', file=output)
        print(f'\thidden_dim: {hidden_dim}', file=output)
        print(f'\tkernel_width: {kernel_width}', file=output)

        contact = ContactCNN(projection_dim, hidden_dim, kernel_width)

        # Create the full model
        use_W = args.use_w
        do_pool = args.do_pool
        pool_width = args.pool_width
        sigmoid = args.sigmoid
        print('# Initializing interaction model with:', file=output)
        if do_pool:
            print(f'\tdo_poool: {do_pool}', file=output)
            print(f'\tpool_width: {pool_width}', file=output)
        print(f'\tuse_w: {use_W}', file=output)
        print(f'\tsigmoid: {sigmoid}',file=output)
        model = ModelInteraction(embedding, contact, use_cuda, use_W = use_W, pool_size=pool_width, do_pool=do_pool, do_sigmoid=sigmoid, use_glider=args.use_glider, glider_map=None, glider_mat=None, glider_param = args.glider_param)

        print(model,file=output)

    else:
        #embedding_size = args.embedding_size
        print('# Loading model from checkpoint {}'.format(args.checkpoint), file=output)
        model = torch.load(args.checkpoint)
        model.use_cuda = use_cuda

    if use_cuda:
        model.cuda()


    # Train the model
    lr = args.lr
    wd = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    report_steps = args.epoch_scale
    inter_weight = args.lambda_
    cmap_weight = 1 - inter_weight
    phat_skew = args.pred_skew
    phat_alpha = args.skew_alpha
    digits = int(np.floor(np.log10(num_epochs))) + 1
    save_prefix = args.save_prefix

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    print(f'# Using save prefix "{save_prefix}"', file=output)
    print(f'# Training with Adam: lr={lr}, weight_decay={wd}', file=output)
    print(f'\tnum_epochs: {num_epochs}', file=output)
    print(f'\tepoch_scale: {report_steps}', file=output)
    print(f'\tbatch_size: {batch_size}', file=output)
    print(f'\tinteraction weight: {inter_weight}', file=output)
    print(f'\tcontact map weight: {cmap_weight}', file=output)
    print(f'\tloss calculation: phat = {phat_alpha}*phat + (1 - {phat_alpha})*(phat^{phat_skew})', file=output)
    output.flush()

    batch_report_fmt = "# [{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}"
    epoch_report_fmt = "# Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"

    N = len(pairs_train_iterator) * batch_size
    for epoch in range(num_epochs):

        model.train()

        n = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        # Train batches
        for (z0, z1, y) in pairs_train_iterator:

            loss, correct, mse, b = interaction_grad(model, z0, z1, y, tensors, use_cuda,
                                                     glider_map = glider_map,
                                                     glider_mat = glider_mat,
                                                     weight=inter_weight,
                                                     skew=phat_skew, alpha = phat_alpha)

            n += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/n

            delta = correct - b*acc_accum
            acc_accum += delta/n

            delta = b*(mse - mse_accum)
            mse_accum += delta/n

            report = ((n - b)//100 < n//100)

            optim.step()
            optim.zero_grad()
            model.clip()

            if report:
                tokens = [epoch+1, num_epochs, n/N, loss_accum, acc_accum, mse_accum]
                print(batch_report_fmt.format(*tokens), file=output)
                output.flush()

        if (epoch + 1) % report_steps == 0:
            model.eval()

            with torch.no_grad():

                inter_loss, inter_correct, inter_mse, inter_pr, inter_re, inter_f1, inter_aupr = interaction_eval(model, pairs_test_iterator, tensors, use_cuda)
                tokens = [epoch+1, num_epochs, inter_loss, inter_correct/(len(pairs_test_iterator)*batch_size), inter_mse, inter_pr, inter_re, inter_f1, inter_aupr]
                print(epoch_report_fmt.format(*tokens), file=output)
                output.flush()

            # Save the model
            if save_prefix is not None:
                save_path = save_prefix + '_epoch' + str(epoch+1).zfill(digits) + '.sav'
                print(f'# Saving model to {save_path}', file=output)
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

        output.flush()

    if save_prefix is not None:
        save_path = save_prefix + '_final.sav'
        print(f'# Saving final model to {save_path}', file=output)
        model.cpu()
        torch.save(model, save_path)
        if use_cuda:
            model.cuda()

def main(args):

    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')

    print(f'# Called as: {" ".join(sys.argv)}',file=output)
    if output is not sys.stdout:
        print(f'Called as: {" ".join(sys.argv)}')

    ## Set the device
    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(f'# Using CUDA device {device} - {torch.cuda.get_device_name(device)}', file=output)
    else:
        print('# Using CPU', file=output)
        device = 'cpu'

    pairs_train_iterator, pairs_test_iterator, tensors, glider_mat, glider_map = load_embeddings_from_args(args, output)
    run_training_from_args(args, pairs_train_iterator, pairs_test_iterator, tensors, glider_mat, glider_map, output)

    output.close()
if __name__ == "__main__":
    main(parse_args())
