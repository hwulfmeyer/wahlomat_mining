import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import wahlomat_lsa2021 as wahlomat_data
from scipy.cluster.hierarchy import _plot_dendrogram, linkage, optimal_leaf_ordering

sns.set(font_scale=0.5)
sns.set_theme(color_codes=True)


metrics = ['euclidean', 'cityblock', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'braycurtis', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
metrics = ['braycurtis', 'canberra', 'cityblock', 'euclidean', 'hamming', 'jaccard']
metrics = ['euclidean']

methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median']
methods = ['average']
"""
single mit cityblock sieht am interessantesten aus
"""

for method in methods:
    for metric in metrics:
        if method in ('centroid', 'median'):
            metric = 'euclidean'
        print(method + " - " + metric)
        row_linkage, col_linkage = (linkage(x, metric=metric, method=method, optimal_ordering=True) for x in (wahlomat_data.df.values, wahlomat_data.df.values.T))
        g = sns.clustermap(wahlomat_data.df, cmap=sns.diverging_palette(22, 145, as_cmap=True), row_linkage=row_linkage, col_linkage=col_linkage, yticklabels=True, linewidths=0.1, dendrogram_ratio=(0.15, 0.15))
        g.cax.set_visible(False)
        #g = sns.clustermap(wahlomat_data.df_t, figsize=(13,6), cmap="vlag_r", metric=metric, method=method)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 8)
        plt.savefig("plots/" + "clustermap_" + method + "_" + metric + ".png", dpi=600)
        #plt.show()
        plt.clf()

pca = PCA()
for n_components in range(1,7,1):
    pca = PCA(n_components=n_components)
    pca.fit_transform(wahlomat_data.df_t)

    pcacomponents = pd.DataFrame(pca.components_, columns=wahlomat_data.df_t.columns, index = ["PCA-" + str(x+1) for x in range(n_components)])
    pcacomponents = pcacomponents.transpose()

    pcacomponents = pcacomponents.sort_values("PCA-1")
    sns.set_theme(color_codes=True)
    if n_components > 1:
        row_linkage, col_linkage = (linkage(x, metric="euclidean", optimal_ordering=True) for x in (pcacomponents.values, pcacomponents.values.T))
        g = sns.clustermap(pcacomponents,  row_cluster=False, col_cluster=False, dendrogram_ratio=(0.4, 0.1), row_linkage=row_linkage, col_linkage=col_linkage, cmap="YlGnBu")
    else:
        g = sns.clustermap(pcacomponents, row_cluster=False, col_cluster=False, dendrogram_ratio=(0.01, 0.01), metric="cityblock", cmap="YlGnBu")
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 8)
    g.cax.set_visible(False)
    plt.savefig("plots/" + "PCA-" + str(n_components) + "_heatmap.png", dpi=600)
    #plt.show()
    plt.clf()

#only plot variance for the largest PCA
exp_var_pca = pca.explained_variance_ratio_

# Cumulative sum of eigenvalues, to create step plot
# for visualizing the variance explained by each principal component
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

# Create the visualization plot
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("plots/" + "PCA-" + str(n_components) + "_explained_variance.png", dpi=600)
plt.show()
plt.clf()