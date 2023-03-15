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
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
from skimage import io, filters
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import preprocessing, mixture

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
    vmin = np.min(im_array)
    vmax = np.max(im_array)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(ax):
        index = 50 if i == 0 else 100 if i == 1 else 150 
        im = axis.imshow(im_array[index, :, :], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axis)
    plt.show()


# %%
def show_vert_slices(im_array):
    vmin = np.min(im_array)
    vmax = np.max(im_array)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(ax):
        if i == 0:
            im = axis.imshow(im_array[z_size//2, :, :], vmin=vmin, vmax=vmax)
        elif i == 1:
            im = axis.imshow(im_array[:, y_size//2, :], vmin=vmin, vmax=vmax)
        elif i == 2:
            im = axis.imshow(im_array[:, :, x_size//2], vmin=vmin, vmax=vmax)
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
sigma = 2

im_alpha_gauss_filtered = filter_im_array(im_alpha, sigma=sigma)
im_beta_gauss_filtered = filter_im_array(im_beta, sigma=sigma)
im_poly_gauss_filtered = filter_im_array(im_poly, sigma=sigma)

show_hor_slices(im_poly_gauss_filtered)
show_vert_slices(im_poly_gauss_filtered)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].hist(im_alpha.flatten(), bins=64, log=True)
ax[0].grid(True)
ax[1].hist(im_beta.flatten(), bins=64, log=True)
ax[1].grid(True)
ax[2].hist(im_poly.flatten(), bins=64, log=True)
ax[2].grid(True)
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].hist(im_alpha_gauss_filtered.flatten(), bins=64, log=True)
ax[0].grid(True)
ax[1].hist(im_beta_gauss_filtered.flatten(), bins=64, log=True)
ax[1].grid(True)
ax[2].hist(im_poly_gauss_filtered.flatten(), bins=64, log=True)
ax[2].grid(True)
plt.show()

# %%
fig, ax = filters.try_all_threshold(im_poly[50])
plt.show()

# %%
thresh = filters.threshold_minimum(im_poly[50])
print(thresh)

mask = im_poly > thresh

show_hor_slices(im_poly)
show_hor_slices(mask)
show_vert_slices(im_poly)
show_vert_slices(mask)

# %%
bins = 128
log = True
slice_number = slice(0, 50)

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].hist(im_alpha[slice_number].flatten(), bins=bins, log=log)
# ax[0].grid(True)
# ax[1].hist(im_beta[slice_number].flatten(), bins=bins, log=log)
# ax[1].grid(True)
# ax[2].hist(im_poly[slice_number].flatten(), bins=bins, log=log)
# ax[2].grid(True)
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].hist(im_alpha[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
ax[0].grid(True)
ax[1].hist(im_beta[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
ax[1].grid(True)
ax[2].hist(im_poly[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
ax[2].grid(True)
plt.show()

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].hist(im_alpha_gauss_filtered[slice_number].flatten(), bins=bins, log=log)
# ax[0].grid(True)
# ax[1].hist(im_beta_gauss_filtered[slice_number].flatten(), bins=bins, log=log)
# ax[1].grid(True)
# ax[2].hist(im_poly_gauss_filtered[slice_number].flatten(), bins=bins, log=log)
# ax[2].grid(True)
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].hist(im_alpha_gauss_filtered[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
ax[0].grid(True)
ax[1].hist(im_beta_gauss_filtered[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
ax[1].grid(True)
ax[2].hist(im_poly_gauss_filtered[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
ax[2].grid(True)
plt.show()

# %%
slice_number = slice(0, 199)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].hist2d(im_alpha_gauss_filtered[slice_number].flatten(), im_beta_gauss_filtered[slice_number].flatten(), bins=64, norm = LogNorm())
ax[1].scatter(im_alpha_gauss_filtered[slice_number].flatten(), im_beta_gauss_filtered[slice_number].flatten(), marker='.')

# %%
slice_number = slice(0, 199)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].hist2d(
    im_alpha_gauss_filtered[slice_number][mask[slice_number]].flatten(), 
    im_beta_gauss_filtered[slice_number][mask[slice_number]].flatten(), 
    bins=64,
    norm = LogNorm()
)
ax[1].scatter(
    im_alpha_gauss_filtered[slice_number][mask[slice_number]].flatten(), 
    im_beta_gauss_filtered[slice_number][mask[slice_number]].flatten(), 
    marker='.'
)

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
train_data = np.array(df)
clf = mixture.GaussianMixture(n_components=3, covariance_type="full")
labels = clf.fit_predict(train_data)

# %%
print(np.unique(labels))
print(f'clf.weights_ {clf.weights_}')
print(f'clf.means_ {clf.means_}')
print(f'clf.covariances_ {clf.covariances_}')

# %%
labeled_df = df.copy()
labeled_df['labels'] = labels
labeled_df.describe()

# %%
rnd_indx = np.random.choice(range(im_poly[mask].size), size=10000, replace=False)
part_df = labeled_df.iloc[rnd_indx]
part_df.describe()

# %%
sns.pairplot(
    part_df, 
    diag_kind="hist", 
    hue='labels', 
    markers='.', 
    palette={-1:'lightgray', 0:'blue', 1:'red', 2:'green' }, 
    diag_kws={'bins':64}
)

# %%
print(f'clf.weights_ {clf.weights_}')
print(f'clf.means_ {clf.means_}')

# %%
poly_colors = im_poly.flatten()
poly_colors[mask.flatten()] = labels
poly_colors[~mask.flatten()] = -1

# %%
palette = np.array([[  0,   0,   0],
                    [  0,   0, 255],
                    [255,   0,   0],
                    [  0, 255,   0]])

im_colors = palette[poly_colors.astype(np.int16)+1]

slices = [50, 100, 120, 135, 150, 175]

images = [im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered]
fig, ax = plt.subplots(len(slices), 4, figsize=(40, len(slices)*10))
for i, slice_number in enumerate(slices):
    for j, image in enumerate(images):
        im = ax[i, j].imshow(image[slice_number], vmin=image.min(), vmax=image.max())
        plt.colorbar(im, ax=ax[i, j])
    ax[i, 3].imshow(im_colors.reshape((*im_alpha_gauss_filtered.shape, 3))[slice_number])

# %%
slice_number = slice(0, 199)

a = im_alpha_gauss_filtered[slice_number][mask[slice_number]].flatten()
b = im_beta_gauss_filtered[slice_number][mask[slice_number]].flatten()
p = im_poly_gauss_filtered[slice_number][mask[slice_number]].flatten()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].hist2d(a, b, bins=64, norm = LogNorm())
# ax[0].plot((0, 1.6678564), (0, 1.19504867), linewidth=2, color='b')
ax[0].plot((0, 1.75), (0, 1.25), linewidth=2, color='b')
# ax[0].axline((0, 0), (1.07490589, 5.21161774), linewidth=2, color='r')
ax[0].axline((0, 0), (3.5, 6), linewidth=2, color='r')
# ax[0].axline((0, 0), (0.62914053, 0.46302298), linewidth=2, color='g')

ax[1].hist2d(a, p, bins=64, norm = LogNorm())
# ax[1].plot((0, 1.6678564), (0, 0.60563116), linewidth=2, color='b')
ax[1].plot((0, 1.75), (0, 0.6), linewidth=2, color='b')
# ax[1].axline((0, 0), (1.07490589, 2.83222735), linewidth=2, color='r')
ax[1].axline((0, 0), (3.5, 4.5), linewidth=2, color='r')
# ax[1].axline((0, 0), (1.07490589, 4.5), linewidth=2, color='r')
# ax[1].axline((0, 0), (0.62914053, 0.66), linewidth=2, color='g')

ax[2].hist2d(b, p, bins=64, norm = LogNorm())
# ax[2].plot((0, 1.19504867), (0, 0.60563116), linewidth=2, color='b')
ax[2].plot((0, 1.25), (0, 0.6), linewidth=2, color='b')
# ax[2].axline((0, 0), (5.21161774, 2.83222735), linewidth=2, color='r')
ax[2].axline((0, 0), (6, 4.5), linewidth=2, color='r')
# ax[2].axline((0, 0), (5.21161774, 4.5), linewidth=2, color='r')
# ax[2].axline((0, 0), (0.46302298, 0.66), linewidth=2, color='g')


# %%
slice_number = slice(0, 199)

a = im_alpha_gauss_filtered[slice_number][mask[slice_number]].flatten()
b = im_beta_gauss_filtered[slice_number][mask[slice_number]].flatten()
p = im_poly_gauss_filtered[slice_number][mask[slice_number]].flatten()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].hist2d(a, b, bins=64, norm = LogNorm())
ax[0].plot((0, clf.means_[0, 0]), (0, clf.means_[0, 1]), linewidth=2, color='b')
ax[0].plot((0, clf.means_[1, 0]), (0, clf.means_[1, 1]), linewidth=2, color='r')
ax[0].plot((0, clf.means_[2, 0]), (0, clf.means_[2, 1]), linewidth=2, color='g')

ax[1].hist2d(a, p, bins=64, norm = LogNorm())
ax[1].plot((0, clf.means_[0, 0]), (0, clf.means_[0, 2]), linewidth=2, color='b')
ax[1].plot((0, clf.means_[1, 0]), (0, clf.means_[1, 2]), linewidth=2, color='r')
ax[1].plot((0, clf.means_[2, 0]), (0, clf.means_[2, 2]), linewidth=2, color='g')

ax[2].hist2d(b, p, bins=64, norm = LogNorm())
ax[2].plot((0, clf.means_[0, 1]), (0, clf.means_[0, 2]), linewidth=2, color='b')
ax[2].plot((0, clf.means_[1, 1]), (0, clf.means_[1, 2]), linewidth=2, color='r')
ax[2].plot((0, clf.means_[2, 1]), (0, clf.means_[2, 2]), linewidth=2, color='g')


# %%
clf.means_

# %%
