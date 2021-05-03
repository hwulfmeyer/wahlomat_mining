import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import wahlomat_data
from scipy.cluster.hierarchy import linkage

sns.set(font_scale=0.5)
sns.set_theme(color_codes=True)


metrics = ['euclidean', 'cityblock', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'braycurtis', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median']

for method in methods:
    for metric in metrics:
        if method in ('centroid', 'median'):
            metric = 'euclidean'
        print(method + " - " + metric)
        row_linkage, col_linkage = (linkage(x, metric=metric, method=method, optimal_ordering=True) for x in (wahlomat_data.df_all_transposed.values, wahlomat_data.df_all_transposed.values.T))
        g = sns.clustermap(wahlomat_data.df_all_transposed, figsize=(13,6), cmap="vlag_r", row_linkage=row_linkage, col_linkage=col_linkage)

        #g = sns.clustermap(wahlomat_data.df_all_transposed, figsize=(13,6), cmap="vlag_r", metric=metric, method=method)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 8)
        plt.savefig("plots/" + "clustermap_" + method + "_" + metric + ".png", dpi=600)
        #plt.show()
        plt.clf()


pca = PCA()
for n_components in range(1,6,1):
    pca = PCA(n_components=n_components)
    model = pca.fit_transform(wahlomat_data.df_all_transposed)

    pcacomponents = pd.DataFrame(pca.components_, columns=wahlomat_data.df_all_transposed.columns,index = ["PCA-" + str(x+1) for x in range(n_components)])

    sns.set_theme(color_codes=True)
    if n_components > 8:
        row_linkage, col_linkage = (linkage(x, metric="euclidean", optimal_ordering=True) for x in (pcacomponents.values, pcacomponents.values.T))
        g = sns.clustermap(pcacomponents,  row_cluster=False, figsize=(13, 3 if n_components == 1 else (n_components+1)), dendrogram_ratio=(0.1, 0.75), row_linkage=row_linkage, col_linkage=col_linkage)
    else:
        g = sns.clustermap(pcacomponents, row_cluster=False, figsize=(13, 3 if n_components == 1 else (n_components+1)), dendrogram_ratio=(0.1, 0.75), metric="euclidean")
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 8)
    plt.savefig("plots/" + "PCA-" + str(n_components) + "_heatmap.png", dpi=600)
    #plt.show()
    plt.clf()

#only plot variance for the largest PCA
exp_var_pca = pca.explained_variance_ratio_

# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
cum_sum_eigenvalues = np.cumsum(exp_var_pca)


# Create the visualization plot
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("plots/" + "PCA-" + str(n_components) + "_explained_variance.png", dpi=600)
#plt.show()
plt.clf()