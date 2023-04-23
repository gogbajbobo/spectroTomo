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
import scipy as sp
from skimage import io, filters
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn import preprocessing, mixture


# %%
def load_tomo_data(sizes, paths):
    spec_tomo_path = '/Users/grimax/Documents/Science/xtomo/spectroTomo/'
    poly_file_path, alpha_file_path, beta_file_path = paths
    im_poly = np.empty(sizes)
    im_alpha = np.empty(sizes)
    im_beta = np.empty(sizes)
    for i in np.arange(sizes[0]):
        poly_path = f'{spec_tomo_path}{poly_file_path}{i:03}.tif'
        alpha_path = f'{spec_tomo_path}{alpha_file_path}{i:03}.tif'
        beta_path = f'{spec_tomo_path}{beta_file_path}{i:03}.tif'
        im_poly[i, :, :] = io.imread(poly_path)
        im_alpha[i, :, :] = io.imread(alpha_path)
        im_beta[i, :, :] = io.imread(beta_path)
    return im_poly, im_alpha, im_beta

exp_0_sizes = 199, 153, 153
pre_path = 'Spectral_tomo_data/'
paths_0 = f'{pre_path}Poly_correct/tomo_poly', f'{pre_path}K-alpha_correct/tomo_k_alpha', f'{pre_path}K-beta_correct/tomo_k_beta'

im_poly_0, im_alpha_0, im_beta_0 = load_tomo_data(exp_0_sizes, paths_0)


print(f'im_poly_0 {im_poly_0.shape}')
print(f'im_alpha_0 {im_alpha_0.shape}')
print(f'im_beta_0 {im_beta_0.shape}')

exp_1_sizes = 301, 150, 150
pre_path = 'rec_spectral_bragg_mar2023/2023_03_27-29_res/'
paths_1 = f'{pre_path}log_poly_fix/rec_poly_', f'{pre_path}log_k-alpha_fix/rec_alpha_', f'{pre_path}log_k-beta_fix/rec_beta_'

im_poly_1, im_alpha_1, im_beta_1 = load_tomo_data(exp_1_sizes, paths_1)

print(f'im_poly_1 {im_poly_1.shape}')
print(f'im_alpha_1 {im_alpha_1.shape}')
print(f'im_beta_1 {im_beta_1.shape}')


# %%
def show_hor_slices(im_array):
    vmin = np.min(im_array)
    vmax = np.max(im_array)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(ax):
        index = 50 if i == 0 else 100 if i == 1 else 150 
        # index = 100 if i == 0 else 175 if i == 1 else 225 
        im = axis.imshow(im_array[index, :, :], vmin=vmin, vmax=vmax)
        # im = axis.imshow(im_array[index, :, :])
        plt.colorbar(im, ax=axis)
    plt.show()
    
    
def show_vert_slices(im_array, vmax=None):
    vmin = np.min(im_array)
    vmax = vmax or np.max(im_array)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    z_size, y_size, x_size = im_array.shape
    for i, axis in enumerate(ax):
        if i == 0:
            im = axis.imshow(im_array[z_size//2, :, :], vmin=vmin, vmax=vmax)
            # im = axis.imshow(im_array[z_size//2, :, :])
        elif i == 1:
            im = axis.imshow(im_array[:, y_size//2, :], vmin=vmin, vmax=vmax)
            # im = axis.imshow(im_array[:, y_size//2, :])
        elif i == 2:
            im = axis.imshow(im_array[:, :, x_size//2], vmin=vmin, vmax=vmax)
            # im = axis.imshow(im_array[:, :, x_size//2])
        plt.colorbar(im, ax=axis)
    plt.show()
    
    
def filter_im_array(im_array, sigma=1):
    filtered_im_array = np.empty(im_array.shape)
    for i, im in enumerate(im_array):
        filtered_im_array[i, :, :] = filters.gaussian(im, sigma=sigma)
    return filtered_im_array


# %%
show_vert_slices(im_poly_0)
show_vert_slices(im_poly_1)

# %%
# h_0_slice = 15
# h_0_slice = 125
h_0_slice = 145
# h_0_slice = 170
h_1_slice = h_0_slice + 70

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# axes[0, 0].imshow(im_poly_0[h_0_slice, :, :])
# axes[0, 1].imshow(im_alpha_0[h_0_slice, :, :])
# axes[0, 2].imshow(im_beta_0[h_0_slice, :, :])

# axes[1, 0].imshow(im_poly_1[h_1_slice, :, :])
# axes[1, 1].imshow(im_alpha_1[h_1_slice, :, :])
# axes[1, 2].imshow(im_beta_1[h_1_slice, :, :])

# plt.show()

sigma = 2

im_poly_gf_0 = filter_im_array(im_poly_0, sigma=sigma)
im_alpha_gf_0 = filter_im_array(im_alpha_0, sigma=sigma)
im_beta_gf_0 = filter_im_array(im_beta_0, sigma=sigma)

im_poly_gf_1 = np.rot90(filter_im_array(im_poly_1, sigma=sigma), axes=(1, 2))
im_alpha_gf_1 = np.rot90(filter_im_array(im_alpha_1, sigma=sigma), axes=(1, 2))
im_beta_gf_1 = np.rot90(filter_im_array(im_beta_1, sigma=sigma), axes=(1, 2))

vmin, vmax = 0, 1

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

im_00 = axes[0, 0].imshow(im_poly_gf_0[h_0_slice, 10:140, 15:145])
im_01 = axes[0, 1].imshow(im_alpha_gf_0[h_0_slice, 10:140, 15:145])
im_02 = axes[0, 2].imshow(im_beta_gf_0[h_0_slice, 10:140, 15:145])
plt.colorbar(im_00, ax=axes[0, 0])
plt.colorbar(im_01, ax=axes[0, 1])
plt.colorbar(im_02, ax=axes[0, 2])

im_10 = axes[1, 0].imshow(im_poly_gf_1[h_1_slice, 7:137, 2:132])
im_11 = axes[1, 1].imshow(im_alpha_gf_1[h_1_slice, 7:137, 3:133])
im_12 = axes[1, 2].imshow(im_beta_gf_1[h_1_slice, 8:138, 9:139])
# im_10 = axes[1, 0].imshow(im_poly_gf_1[h_1_slice, 70:90, 25:50])
# im_11 = axes[1, 1].imshow(im_alpha_gf_1[h_1_slice, 70:90, 26:51])
# im_12 = axes[1, 2].imshow(im_beta_gf_1[h_1_slice, 71:91, 32:57])
plt.colorbar(im_10, ax=axes[1, 0])
plt.colorbar(im_11, ax=axes[1, 1])
plt.colorbar(im_12, ax=axes[1, 2])

plt.show()


# %%
im_poly_gf_0 = im_poly_gf_0[:, 10:140, 15:145]
im_alpha_gf_0 = im_alpha_gf_0[:, 10:140, 15:145]
im_beta_gf_0 = im_beta_gf_0[:, 10:140, 15:145]

im_poly_gf_1 = im_poly_gf_1[:, 7:137, 2:132]
im_alpha_gf_1 = im_alpha_gf_1[:, 7:137, 3:133]
im_beta_gf_1 = im_beta_gf_1[:, 8:138, 9:139]

print(f'im_poly_gf_0 {im_poly_gf_0.shape}')
print(f'im_alpha_gf_0 {im_alpha_gf_0.shape}')
print(f'im_beta_gf_0 {im_beta_gf_0.shape}')

print(f'im_poly_gf_1 {im_poly_gf_1.shape}')
print(f'im_alpha_gf_1 {im_alpha_gf_1.shape}')
print(f'im_beta_gf_1 {im_beta_gf_1.shape}')

# %%
slices_y_0 = slice(20, 100)
slices_x_0 = slice(20, 100)

im_poly_sliced_0 = im_poly_gf_0[:, slices_y_0, slices_x_0]
im_alpha_sliced_0 = im_alpha_gf_0[:, slices_y_0, slices_x_0]
im_beta_sliced_0 = im_beta_gf_0[:, slices_y_0, slices_x_0]

slices_z_1 = slice(70, 301)
slices_y_1 = slice(23, 103)
slices_y_1_b = slice(22, 102)
slices_x_1 = slice(28, 108)
slices_x_1_a = slice(27, 107)
slices_x_1_b = slice(21, 101)

im_poly_sliced_1 = im_poly_gf_1[slices_z_1, slices_y_1, slices_x_1]
im_alpha_sliced_1 = im_alpha_gf_1[slices_z_1, slices_y_1, slices_x_1_a]
im_beta_sliced_1 = im_beta_gf_1[slices_z_1, slices_y_1_b, slices_x_1_b]

print(f'im_poly sliced 0 {im_poly_sliced_0.shape}')
print(f'im_alpha sliced 0 {im_alpha_sliced_0.shape}')
print(f'im_beta sliced 0 {im_beta_sliced_0.shape}')

print(f'im_poly sliced 1 {im_poly_sliced_1.shape}')
print(f'im_alpha sliced 1 {im_alpha_sliced_1.shape}')
print(f'im_beta sliced 1 {im_beta_sliced_1.shape}')

show_vert_slices(im_poly_sliced_0)
show_vert_slices(im_poly_sliced_1)


# %%
im_poly_gf_0 = np.copy(im_poly_sliced_0)
im_alpha_gf_0 = np.copy(im_alpha_sliced_0)
im_beta_gf_0 = np.copy(im_beta_sliced_0)

im_poly_gf_1 = np.copy(im_poly_sliced_1)
im_alpha_gf_1 = np.copy(im_alpha_sliced_1)
im_beta_gf_1 = np.copy(im_beta_sliced_1)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(im_poly_gf_0[145, :, :])
axes[0, 1].imshow(im_poly_gf_0[:, :, 35])
axes[1, 0].imshow(im_poly_gf_1[145, :, :])
axes[1, 1].imshow(im_poly_gf_1[:, :, 35])
plt.show()


# %%
def show_all_maps(
    im_a, im_b, im_p, 
    a_lim=None, b_lim=None, p_lim=None, 
    mask=None,
    slice_number=None, 
    bins=128,
    figsize=(40, 10),
):
        
    a_lim = a_lim or [np.min(im_a), np.max(im_a)]
    b_lim = b_lim or [np.min(im_b), np.max(im_b)]
    p_lim = p_lim or [np.min(im_p), np.max(im_p)]
    slice_number = slice_number or slice(0, im_a.shape[0])
    if mask is None:
        mask = np.full(im_a.shape, True)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    _im_a = im_a[slice_number][mask[slice_number]].flatten()
    _im_b = im_b[slice_number][mask[slice_number]].flatten()
    _im_p = im_p[slice_number][mask[slice_number]].flatten()
    
    h, xedges, yedges, im = ax[0].hist2d(_im_a, _im_b, bins=bins, norm = LogNorm())
    ax[0].set_xlim(a_lim)
    ax[0].set_ylim(b_lim)
    ax[0].grid(True)
    ax[0].tick_params(labelsize=22)
    ax[0].set_xlabel('MoK⍺', fontsize=36)
    ax[0].set_ylabel('MoKβ', fontsize=36)
    cbar = plt.colorbar(im, ax=ax[0])
    cbar.ax.tick_params(labelsize=18) 

    h, xedges, yedges, im = ax[1].hist2d(_im_a, _im_p, bins=bins, norm = LogNorm())
    ax[1].set_xlim(a_lim)
    ax[1].set_ylim(p_lim)
    ax[1].grid(True)
    ax[1].tick_params(labelsize=22)
    ax[1].set_xlabel('MoK⍺', fontsize=36)
    ax[1].set_ylabel('Poly', fontsize=36)
    cbar = plt.colorbar(im, ax=ax[1])
    cbar.ax.tick_params(labelsize=18) 

    h, xedges, yedges, _im = ax[2].hist2d(_im_b, _im_p, bins=bins, norm = LogNorm())
    ax[2].set_xlim(b_lim)
    ax[2].set_ylim(p_lim)
    ax[2].grid(True)
    ax[2].tick_params(labelsize=22)
    ax[2].set_xlabel('MoKβ', fontsize=36)
    ax[2].set_ylabel('Poly', fontsize=36)
    cbar = plt.colorbar(im, ax=ax[2])
    cbar.ax.tick_params(labelsize=18) 
    
    plt.show()
    # return ax


# %%
a_lim = [-.5, 4]
b_lim = [-1, 10]
p_lim = [-.1, 5]

show_all_maps(im_alpha_gf_0, im_beta_gf_0, im_poly_gf_0, a_lim=a_lim, b_lim=b_lim, p_lim=p_lim)
show_all_maps(im_alpha_gf_1, im_beta_gf_1, im_poly_gf_1, a_lim=a_lim, b_lim=b_lim, p_lim=p_lim)

# %%
data_0 = {
    'alpha': im_alpha_gf_0.flatten(), 
    'beta': im_beta_gf_0.flatten(), 
    'poly': im_poly_gf_0.flatten(),
}
df_0 = pd.DataFrame(data=data_0)
df_0.describe()

# %%
clf = mixture.BayesianGaussianMixture(n_components=3)
labels_0 = clf.fit_predict(df_0)

print(np.unique(labels_0))
print(f'clf.weights_ {clf.weights_}')
print(f'clf.means_ {clf.means_}')

# %%
df_0['labels'] = labels_0
df_0.describe()

# %%
rnd_indx = np.random.choice(range(im_alpha_gf_0.size), size=10000, replace=False)
part_df_0 = df_0.iloc[rnd_indx]
part_df_0.describe()

# %%
# def hide_current_axis(*args, **kwds):
#     plt.gca().set_visible(False)

g = sns.PairGrid(
    part_df_0, 
    hue='labels',
    palette={0: 'blue', 1: 'red', 2: 'green'},
    corner=True,
    # diag_sharey=False,
)
g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot, marker='.')
g.map_lower(sns.scatterplot, marker='.')
# g.map_upper(hide_current_axis)
g.fig.legend(handles=g._legend_data.values(), labels=['NaCl', 'LiNbO3', 'Ag begenat'], bbox_to_anchor=(.85, .75))


# %%
# data_1 = {
#     'alpha': im_alpha_gf_1[:199, :, :].flatten(), 
#     'beta': im_beta_gf_1[:199, :, :].flatten(), 
#     'poly': im_poly_gf_1[:199, :, :].flatten(),
# }
data_1 = {
    'alpha': im_alpha_gf_1.flatten(), 
    'beta': im_beta_gf_1.flatten(), 
    'poly': im_poly_gf_1.flatten(),
}
df_1 = pd.DataFrame(data=data_1)
df_1.describe()

# %%
clf = mixture.BayesianGaussianMixture(n_components=3)
labels_1 = clf.fit_predict(df_1)

print(np.unique(labels_1))
print(f'clf.weights_ {clf.weights_}')
print(f'clf.means_ {clf.means_}')

# %%
df_1['labels'] = labels_1
df_1.describe()

# %%
rnd_indx = np.random.choice(range(df_1[['alpha']].size), size=10000, replace=False)
part_df_1 = df_1.iloc[rnd_indx]
part_df_1.describe()

# %%
g = sns.PairGrid(
    part_df_1, 
    hue='labels',
    palette={0: 'blue', 1: 'red', 2: 'green'},
    corner=True,
)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot, marker='.')
g.fig.legend(handles=g._legend_data.values(), labels=['NaCl', 'LiNbO3', 'Ag begenat'], bbox_to_anchor=(.85, .75))


# %%
poly_colors_0 = np.copy(labels_0)

# %%
palette = np.array([[  0,   0,   0],
                    [  0,   0, 255],
                    [255,   0,   0],
                    [  0, 255,   0]])

slices = [100, 125, 150, 175]

images = [im_alpha_gf_0, im_beta_gf_0, im_poly_gf_0]
im_colors_0 = palette[poly_colors_0.astype(np.int16)+1].reshape((*images[0].shape, 3))

fig, ax = plt.subplots(len(slices), 4, figsize=(20, len(slices)*5))
for i, slice_number in enumerate(slices):
    for j, image in enumerate(images):
        # im = ax[i, j].imshow(image[slice_number], vmin=image.min(), vmax=image.max())
        im = ax[i, j].imshow(image[slice_number], vmin=0, vmax=4)
        # im = ax[i, j].imshow(image[slice_number])
        plt.colorbar(im, ax=ax[i, j])
    ax[i, 3].imshow(im_colors_0[slice_number])
    plt.colorbar(im, ax=ax[i, 3])
plt.show()

# %%
fig, ax = plt.subplots(2, 2, figsize=(7, 10))

im = ax[0, 0].imshow(im_poly_gf_0[:, :, 35], vmin=0, vmax=2.5)
# im = ax[0, 0].imshow(im_poly_gf_0[:, :, 35])
plt.colorbar(im, ax=ax[0, 0])
ax[0, 1].imshow(im_colors_0[:, :, 35])

im = ax[1, 0].imshow(im_poly_gf_0[:, 35, :], vmin=0, vmax=2.5)
# im = ax[1, 0].imshow(im_poly_gf_0[:, 35, :])
plt.colorbar(im, ax=ax[1, 0])
ax[1, 1].imshow(im_colors_0[:, 35, :])

plt.show()

# %%
poly_colors_1 = np.copy(labels_1)

# %%
slices = [100, 125, 150, 175]

images = [im_alpha_gf_1, im_beta_gf_1, im_poly_gf_1]
im_colors_1 = palette[poly_colors_1.astype(np.int16)+1].reshape((*images[0].shape, 3))

fig, ax = plt.subplots(len(slices), 4, figsize=(20, len(slices)*5))
for i, slice_number in enumerate(slices):
    for j, image in enumerate(images):
        # im = ax[i, j].imshow(image[slice_number], vmin=image.min(), vmax=image.max())
        # im = ax[i, j].imshow(image[slice_number], vmin=0, vmax=6)
        im = ax[i, j].imshow(image[slice_number])
        plt.colorbar(im, ax=ax[i, j])
    ax[i, 3].imshow(im_colors_1[slice_number])
    plt.colorbar(im, ax=ax[i, 3])
plt.show()

# %%
fig, ax = plt.subplots(2, 2, figsize=(7, 10))

# im = ax[0, 0].imshow(im_poly_gf_1[:, :, 35], vmin=image.min(), vmax=image.max())
im = ax[0, 0].imshow(im_poly_gf_1[:, :, 35])
plt.colorbar(im, ax=ax[0, 0])
ax[0, 1].imshow(im_colors_1[:, :, 35])

# im = ax[1, 0].imshow(im_poly_gf_1[:, 35, :], vmin=image.min(), vmax=image.max())
im = ax[1, 0].imshow(im_poly_gf_1[:, 35, :])
plt.colorbar(im, ax=ax[1, 0])
ax[1, 1].imshow(im_colors_1[:, 35, :])

plt.show()

# %%

# %%
