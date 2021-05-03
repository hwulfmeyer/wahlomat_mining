import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import wahlomat_data

sns.set(font_scale=0.5)
sns.set_theme(color_codes=True)
g = sns.clustermap(wahlomat_data.df_all_transposed, figsize=(12,6), cmap="vlag_r", metric="correlation")
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 8)
plt.show()

pca = PCA()

"""
pca.fit(df_all_transposed)
X_pca = pca.transform(df_all_transposed)"""

for n_components in range(1,5,1):
    pca = PCA(n_components=n_components)
    model = pca.fit_transform(wahlomat_data.df_all_transposed)
    exp_var_pca = pca.explained_variance_ratio_

    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    print(pca.components_)
    #
    # Create the visualization plot
    #
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    pcacomponents = pd.DataFrame(pca.components_,columns=wahlomat_data.df_all_transposed.columns,index = ["PCA-" + str(x+1) for x in range(n_components)])

    sns.set(font_scale=0.5)
    sns.set_theme(color_codes=True)
    g = sns.clustermap(pcacomponents, row_cluster=False,figsize=(12,6), metric="euclidean")
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 8)
    plt.show()