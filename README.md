# UMAP and Phenograph projection

# About
This pipeline is created to project single cell RNA sequencing profiles from one dataset to another based on Spearman's correlation as a distance. This is useful for multi dataset integration. 

# Dependencies

- Python 3.6 or higher
- Pandas
- Seaborn
- Scikit-learn
- UMAP (https://github.com/lmcinnes/umap)
- Phenograph (https://github.com/dpeerlab/PhenoGraph)

# Usage:

1. UMAP projection

`python umap_transform.py -rm REFDATA -pm QUERY1 QUERY2 -p PREFIX -m marker_genes.txt -k 20`

2. Phenograph projecction

`python phenograph_transform.py -rm REFDATA -pm QUERY1 QUERY2 -p PREFIX -m marker_genes.txt -k 20`


where REFDATA can be a loom file or a tab-delimited matrix of molecular counts for the reference (first two columns contain GIDS and gene symbols, subsequent column contain counts for each cell); QUERY can be a loom file or a matrix of molecular counts for query sample; markers_genes.txt is a tab-delimited file, where the first column contains GIDS for computing similarity (usually highly variable genes). There should be no header in any of the files.

