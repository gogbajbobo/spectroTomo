# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import preprocessing
from matplotlib.colors import LogNorm, ListedColormap


# %%
z_size, y_size, x_size = 199, 153, 153
im_poly = np.empty((z_size, y_size, x_size))
im_alpha = np.empty((z_size, y_size, x_size))
im_beta = np.empty((z_size, y_size, x_size))

for i in np.arange(199):

    poly_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/Spectral_tomo_data/Poly_correct/tomo_poly{i:03}.tif'
    alpha_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/Spectral_tomo_data/K-alpha_correct/tomo_k_alpha{i:03}.tif'
    beta_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/Spectral_tomo_data/K-beta_correct/tomo_k_beta{i:03}.tif'

    im_poly[i, :, :] = io.imread(poly_path)
    im_alpha[i, :, :] = io.imread(alpha_path)
    im_beta[i, :, :] = io.imread(beta_path)

print(f'im_poly {im_poly.shape}')
print(f'im_alpha {im_alpha.shape}')
print(f'im_beta {im_beta.shape}')


# %%
def show_hor_slices(im_array):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(ax):
        index = 50 if i == 0 else 100 if i == 1 else 150 
        im = axis.imshow(im_array[index, :, :])
        plt.colorbar(im, ax=axis)
    plt.show()


# %%
def show_vert_slices(im_array):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(ax):
        if i == 0:
            im = axis.imshow(im_array[z_size//2, :, :])
        elif i == 1:
            im = axis.imshow(im_array[:, y_size//2, :])
        elif i == 2:
            im = axis.imshow(im_array[:, :, x_size//2])
        plt.colorbar(im, ax=axis)
    plt.show()


# %%
def filter_im_array(im_array, sigma=1):
    filtered_im_array = np.empty(im_array.shape)
    for i, im in enumerate(im_array):
        filtered_im_array[i, :, :] = filters.gaussian(im, sigma=sigma)
    return filtered_im_array

# gauss_filtered_im_array = filter_im_array(im_poly)

show_hor_slices(im_poly)
# show_hor_slices(gauss_filtered_im_array)

show_vert_slices(im_poly)
# show_vert_slices(gauss_filtered_im_array)

# %%
fig, ax = filters.try_all_threshold(im_poly[50])
plt.show()

# %%
thresh = filters.threshold_minimum(im_poly[50])
print(thresh)

show_hor_slices(im_poly)
show_hor_slices(im_poly > thresh)
show_vert_slices(im_poly)
show_vert_slices(im_poly > thresh)

# %%
slice = 120
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].hist2d(im_alpha[slice].flatten(), im_beta[slice].flatten(), bins=64, norm = LogNorm())
ax[1].scatter(im_alpha[slice].flatten(), im_beta[slice].flatten(), marker='.')

# %%
mask = im_poly > thresh

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].hist2d(im_alpha[slice][mask[slice]].flatten(), im_beta[slice][mask[slice]].flatten(), bins=64)
ax[1].scatter(im_alpha[slice][mask[slice]].flatten(), im_beta[slice][mask[slice]].flatten(), marker='.')

# %%
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].hist2d(im_alpha[mask].flatten(), im_beta[mask].flatten(), bins=64, norm = LogNorm())
ax[1].scatter(im_alpha[mask].flatten(), im_beta[mask].flatten(), marker='.')

# %%
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(xs=im_alpha[mask].flatten(), ys=im_beta[mask].flatten(), zs=im_poly[mask].flatten(), marker='.')
# ax.set_xlabel('Alpha')
# ax.set_ylabel('Beta')
# ax.set_zlabel('Poly')

# plt.show()

# %%
data = {'alpha': im_alpha[mask].flatten(), 'beta': im_beta[mask].flatten(), 'poly': im_poly[mask].flatten()}
df = pd.DataFrame(data=data)
df.describe()

# %%
sns.pairplot(df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})

# %%
slice = 120
im_a = im_alpha[slice][mask[slice]].flatten()
im_b = im_beta[slice][mask[slice]].flatten()
im_p = im_poly[slice][mask[slice]].flatten()

data = {'alpha': im_a, 'beta': im_b, 'poly': im_p}
df = pd.DataFrame(data=data)
df.describe()

# %%
sns.pairplot(df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})

# %%
sigma = 2
im_alpha_gauss_filtered = filter_im_array(im_alpha, sigma=sigma)
im_beta_gauss_filtered = filter_im_array(im_beta, sigma=sigma)
im_poly_gauss_filtered = filter_im_array(im_poly, sigma=sigma)

# %%
data = {
    'alpha gauss filter': im_alpha_gauss_filtered[mask].flatten(), 
    'beta gauss filter': im_beta_gauss_filtered[mask].flatten(), 
    'poly gauss filter': im_poly_gauss_filtered[mask].flatten(),
}
df = pd.DataFrame(data=data)
print(df.describe())

sns.pairplot(df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})

# %%
slice = 120
im_a_f = im_alpha_gauss_filtered[slice][mask[slice]].flatten()
im_b_f = im_beta_gauss_filtered[slice][mask[slice]].flatten()
im_p_f = im_poly_gauss_filtered[slice][mask[slice]].flatten()

data = {'alpha gauss filter': im_a_f, 'beta gauss filter': im_b_f, 'poly gauss filter': im_p_f}
df = pd.DataFrame(data=data)
print(df.describe())

sns.pairplot(df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})


# %%
rnd_indx = np.random.choice(range(im_poly[mask].size), size=10000, replace=False)
rnd_indx

# %%
data = {
    'alpha gauss filter': im_alpha_gauss_filtered[mask].flatten()[rnd_indx], 
    'beta gauss filter': im_beta_gauss_filtered[mask].flatten()[rnd_indx], 
    'poly gauss filter': im_poly_gauss_filtered[mask].flatten()[rnd_indx],
}
df = pd.DataFrame(data=data)
print(df.describe())

sns.pairplot(df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})


# %%
data = {
    'alpha gauss filter kde': im_alpha_gauss_filtered[mask].flatten()[rnd_indx], 
    'beta gauss filter kde': im_beta_gauss_filtered[mask].flatten()[rnd_indx], 
    'poly gauss filter kde': im_poly_gauss_filtered[mask].flatten()[rnd_indx],
}
df = pd.DataFrame(data=data)
print(df.describe())

sns.pairplot(df, kind="kde", diag_kind='hist', plot_kws={'fill':True,'cbar':True}, diag_kws={'bins':64})

# %%
data = {
    'alpha gauss filter': im_alpha_gauss_filtered[mask].flatten()[rnd_indx], 
    'beta gauss filter': im_beta_gauss_filtered[mask].flatten()[rnd_indx], 
    'poly gauss filter': im_poly_gauss_filtered[mask].flatten()[rnd_indx],
}
df = pd.DataFrame(data=data)

scaler = preprocessing.StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

print(df.describe())

scaled_df = pd.DataFrame(data=scaled_data, columns=['alpha filtered scaled', 'beta filtered scaled', 'poly filtered scaled'])
print(scaled_df.describe())

# %%
sns.pairplot(scaled_df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})

# %%
clustering = DBSCAN(eps=0.1, min_samples=20).fit(scaled_df)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
print(set(labels))

plt.hist(labels, bins=len(set(labels)))

# %%
labeled_df = scaled_df.copy()
labeled_df['labels'] = labels
labeled_df.describe()

# %%
sns.pairplot(
    labeled_df, 
    diag_kind="hist", 
    hue='labels', 
    markers='.', 
    palette={-1:'lightgray', 0:'blue', 1:'red' }, 
    diag_kws={'bins':64}
)

# %%
print(labels)
print(np.unique(labels).size)
n_clusters_ = np.unique(labels).size - (1 if -1 in labels else 0)
print(n_clusters_)

print(n_clusters_ == 2 or n_clusters_ == 3)

clusters, counts = np.unique(labels, return_counts=True)
print(clusters, counts)
# counts[np.where(clusters > -1)[0]][0]

labels[labels != -1].size / labels[labels == -1].size


# %%
def DBSCAN_score(estimator, X):
    print('\n')
    eps = estimator.get_params()['eps']
    min_samples = estimator.get_params()['min_samples']
    print(f'eps {eps}')
    print(f'min_samples {min_samples}')
    print(f'X size: {X.shape}')
    model = estimator.fit(X)
    labels = model.labels_
    n_clusters_ = np.unique(labels).size - (1 if -1 in labels else 0)
    print(n_clusters_)
    print(np.unique(labels))
    print(np.histogram(labels, bins=np.unique(labels).size))
    if n_clusters_ < 2 or n_clusters_ > 3:
        print(-1)
        return -1
    noise_size = labels[labels == -1].size
    score = n_clusters_ if noise_size == 0 else (labels[labels != -1].size / noise_size)
    print(score)
    return score

# eps=0.1, min_samples=20
param_grid = {
    "eps": np.arange(0.08, 0.13, step=0.02),
    "min_samples": np.arange(16, 24, step=2),
}
cv = ShuffleSplit(test_size=9999, n_splits=1)
grid_search = GridSearchCV(
    DBSCAN(), param_grid=param_grid, scoring=DBSCAN_score, cv=cv
)
grid_search.fit(scaled_df)

grid_search.cv_results_

# %%
df = pd.DataFrame(grid_search.cv_results_)[
    ["param_eps", "param_min_samples", "mean_test_score"]
]
df.sort_values(by="mean_test_score")

# %%
clustering = DBSCAN(eps=0.1, min_samples=18).fit(scaled_df)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
print(set(labels))

plt.hist(labels, bins=len(set(labels)))

# %%
labeled_df = scaled_df.copy()
labeled_df['labels'] = labels
labeled_df.describe()

# %%
sns.pairplot(
    labeled_df, 
    diag_kind="hist", 
    hue='labels', 
    markers='.', 
    palette={-1:'lightgray', 0:'blue', 1:'red', 2:'green' }, 
    diag_kws={'bins':64}
)

# %%
