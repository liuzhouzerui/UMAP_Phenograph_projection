#! /usr/bin/python
import numpy as np
import argparse
import anndata
import pandas as pd
import phenograph
from clusterdiffex.cluster import run_phenograph_approx_knn
from clusterdiffex.distance import spearmanr
from sklearn.neighbors import NearestNeighbors
from phenograph.classify import preprocess, random_walk_probabilities
from phenograph.core import neighbor_graph, jaccard_kernel

def parser_user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rm','--ref-matrix',required=True,help='Path to count matrix for reference.')
    parser.add_argument('-pm','--proj-matrices',nargs='+',required=True,help='Path to count matrices for query samples.')
    parser.add_argument('-o','--output',required=True,help='dir for output including path.')
    parser.add_argument('-p','--prefix',required=True,help='Prefix for output including path.')
    parser.add_argument('-m','--markers',required=True,help='Path to file with one-column list of marker gids (cell-cell similarity will be computed using these markers).')
    parser.add_argument('-k','-k',required=True,default=20,type=int,help='Integer value of k parameter for UMAP.')
    return parser

# for loading molecular count matrix for a list of marker gids with format GID\tSYMBOL\tCTS_CELL1\tCTS_CELL2\t...
# first column in marker_INFILE contains list of marker gids
def load_marker_matrix(matrix_INFILE,marker_INFILE):
    if matrix_INFILE.split('.')[-1] == 'txt':
        gids = []
        genes = []
        matrix = []
        with open(marker_INFILE) as f:
            markers = set([line.split()[0] for line in f])
        with open(matrix_INFILE) as f:
            for line in f:
                llist = line.split()
                gid = llist[0]
                if gid in markers:
                    gids.append(gid)
                    genes.append(llist[1])
                    try:
                        matrix.append([int(pt) for pt in llist[2::]])
                    except ValueError:
                        matrix.append([float(pt) for pt in llist[2::]])
        gids = np.array(gids)
        ind = np.argsort(gids)
        gids = gids[ind]
        genes = np.array(genes)[ind]
        matrix = np.array(matrix)[ind]
    if matrix_INFILE.split('.')[-1] == 'loom':
        loom = anndata.read_loom(matrix_INFILE, obs_names='', var_names='')
        markers = pd.read_csv(marker_INFILE, sep='\t', header=None)
        loom_filter = loom.T[np.isin(loom.T.obs['Accession'], markers[0])].copy().T
        gids = loom_filter.var['Accession'].values
        genes = loom_filter.var['Gene'].values
        matrix = loom_filter.X.todense().T
    return gids, genes, matrix


parser = parser_user_input()
ui = parser.parse_args()

print('Loading data...') # get reference count matrix (only marker genes)
rgids,rgenes,rmatrix = load_marker_matrix(ui.ref_matrix,ui.markers)

# get cluster of the reference
clusterfile = ui.output+'/'+ui.prefix+'.pg.txt'
communities = pd.read_csv(clusterfile, sep='\t',header=None,dtype='int')[0].to_numpy()

# reformat training data
train = []
for i in range(communities.max()+1):
    matrix = rmatrix.T[communities==i]
    train.append(matrix)

def create_graph_spearmanr(data, k=20, metric=spearmanr, n_jobs=-1,algorithm='auto'):
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,  # k+1 because results include self
        n_jobs=n_jobs,  # use multiple cores if possible
        metric=metric,  # primary metric
        algorithm=algorithm,  # kd_tree is fastest for minkowski metrics
    ).fit(data)
    d, idx = nbrs.kneighbors(data)

    # Remove self-distances if these are in fact included
    if idx[0, 0] == 0:
        idx = np.delete(idx, 0, axis=1)
        d = np.delete(d, 0, axis=1)
    else:  # Otherwise delete the _last_ column of d and idx
        idx = np.delete(idx, -1, axis=1)
        d = np.delete(d, -1, axis=1)
    return d, idx

def classification(train, test, metric=spearmanr, k=20):
    data, labels = preprocess(train, test)
    d, idx = create_graph_spearmanr(data, ui.k, metric=spearmanr)
    A = neighbor_graph(jaccard_kernel, {"idx": idx})
    P = random_walk_probabilities(A, labels)
    c = np.argmax(P, axis=1)
    return c, P

for i,proj_matrix in enumerate(ui.proj_matrices):
    pgids,pgenes,pmatrix = load_marker_matrix(proj_matrix,ui.markers) # get query count matrix (only marker genes)
    if len(pgids) < len(rgids):
        print('Error: Some marker GIDS in the reference matrix are missing in the query matrix %(i)d.' % vars())
        exit()
    elif len(rgids) > len(pgids):
        print('Error: Some marker GIDS in query matrix %(i)d are missing from the reference matrix.' % vars())
        exit()
    c, P = classification(train, pmatrix.T, metric=spearmanr, k=ui.k)
    c_output = ui.output+'/'+ui.prefix+'.proj.'+str(i)+'.pg.txt'
    np.savetxt(c_output, c, fmt='%i')
    p_output = ui.output+'/'+ui.prefix+'.proj.'+str(i)+'.pg.probability.txt'
    pd.DataFrame(P).to_csv(p_output,sep='\t', header=False, index=False) # write projection coordinates to file




