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
thresh = 0.25

print(thresh)

mask = im_poly > thresh

show_hor_slices(im_poly)
show_hor_slices(mask)
show_vert_slices(im_poly)
show_vert_slices(mask)

# %%
bins = 128
log = True
slice_number = slice(190, 199)

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
def show_maps(im_a, im_b, xlim=None, ylim=None, mask=None):
    
    xlim = xlim or [np.min(im_a), np.max(im_a)]
    ylim = ylim or [np.min(im_b), np.max(im_b)]

    for i in np.arange(4):

        slice_number = slice(i*50, (i+1)*50-1)

        alpha = (im_a[slice_number] if mask is None else im_a[slice_number][mask[slice_number]]).flatten()
        beta = (im_b[slice_number] if mask is None else im_b[slice_number][mask[slice_number]]).flatten()

        rnd_indx = np.random.choice(range(alpha.size), size=5000, replace=False)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist2d(alpha, beta, bins=64, norm = LogNorm())
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        ax[1].scatter(alpha[rnd_indx], beta[rnd_indx], marker='.')
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)

        plt.show()


def show_full_maps(im_a, im_b, xlim=None, ylim=None, mask=None):
    
    xlim = xlim or [np.min(im_a), np.max(im_a)]
    ylim = ylim or [np.min(im_b), np.max(im_b)]

    slice_number = slice(0, 199)

    alpha = (im_a[slice_number] if mask is None else im_a[slice_number][mask[slice_number]]).flatten()
    beta = (im_b[slice_number] if mask is None else im_b[slice_number][mask[slice_number]]).flatten()

    rnd_indx = np.random.choice(range(alpha.size), size=5000, replace=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist2d(alpha, beta, bins=64, norm = LogNorm())
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].scatter(alpha[rnd_indx], beta[rnd_indx], marker='.')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)

    plt.show()

def show_sliced_maps(im_a, im_b, xlim=None, ylim=None, mask=None, slice_number = slice(0, 199)):
    
    xlim = xlim or [np.min(im_a), np.max(im_a)]
    ylim = ylim or [np.min(im_b), np.max(im_b)]

    alpha = (im_a[slice_number] if mask is None else im_a[slice_number][mask[slice_number]]).flatten()
    beta = (im_b[slice_number] if mask is None else im_b[slice_number][mask[slice_number]]).flatten()

    size = alpha.size if alpha.size < 5000 else 5000
    rnd_indx = np.random.choice(range(alpha.size), size=size, replace=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist2d(alpha, beta, bins=64, norm = LogNorm())
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].scatter(alpha[rnd_indx], beta[rnd_indx], marker='.')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)

    plt.show()

def show_sliced_map(
    im1, im2, 
    xlim=None, ylim=None, 
    mask=None, 
    slice_number=slice(0, 199), 
    x_label=None, y_label=None
):
    
    xlim = xlim or [np.min(im1), np.max(im1)]
    ylim = ylim or [np.min(im2), np.max(im2)]

    image1 = (im1[slice_number] if mask is None else im1[slice_number][mask[slice_number]]).flatten()
    image2 = (im2[slice_number] if mask is None else im2[slice_number][mask[slice_number]]).flatten()

    size = image1.size if image1.size < 5000 else 5000
    rnd_indx = np.random.choice(range(image1.size), size=size, replace=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist2d(image1, image2, bins=64, norm = LogNorm())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if x_label:
        ax.set_xlabel(x_label, fontsize=18)
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)

    plt.show()
    


# %%
show_sliced_map(
    im_alpha_gauss_filtered, 
    im_beta_gauss_filtered, 
    [-0.5, 4], [-1, 6.5], 
    mask=mask,
    #slice_number = slice(190, 199),
    x_label = 'MoK⍺',
    y_label = 'MoKβ',
)

# %%
show_sliced_map(
    im_alpha_gauss_filtered, im_poly_gauss_filtered, 
    [-0.5, 4], [-0.5, 5], 
    mask=mask,
    # slice_number = slice(190, 199),
    x_label = 'MoK⍺',
    y_label = 'Polychromatic',
)

# %%
show_sliced_map(
    im_beta_gauss_filtered, im_poly_gauss_filtered, 
    [-1, 6.5], [-0.5, 5], 
    mask=mask,
    # slice_number = slice(190, 199),
    x_label = 'MoKβ',
    y_label = 'Polychromatic',
)

# %%
show_sliced_map(im_alpha, im_beta, [-0.5, 1], [-1.5, 3], slice_number = slice(190, 199))

# %%
show_sliced_map(
    im_alpha, im_poly, 
    [-0.5, 1], [-0.2, 1.2], slice_number = slice(190, 199)
)

# %%
show_sliced_map(
    im_beta, im_poly, 
    [-1.5, 3], [-0.2, 1.2], slice_number = slice(190, 199)
)

# %%
show_sliced_maps(im_alpha_gauss_filtered, im_poly_gauss_filtered, [-0.5, 1], [-0.5, 1], mask=~mask, slice_number = slice(190, 199))

# %%
show_sliced_maps(im_alpha_gauss_filtered, im_poly_gauss_filtered, [-0.5, 1], [-0.5, 1], mask=mask, slice_number = slice(190, 199))

# %%
show_maps(im_alpha, im_beta, [-1, 4], [-10, 10])

# %%
xlim = [-1, 5]
ylim = [-5, 8]
# limits taken from all images

# %%
show_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim)

# %%
show_full_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim)

# %%
show_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim, mask)

# %%
show_full_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim, mask)

# %%
z_size, y_size, x_size = 199, 150, 150
im_poly = np.empty((z_size, y_size, x_size))
im_alpha = np.empty((z_size, y_size, x_size))
im_beta = np.empty((z_size, y_size, x_size))


for i in np.arange(199):

    poly_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/rec_spectral_feb2023_fix/center/Poly_center_fix/rec_poly_center{i:03}.tif'
    alpha_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/rec_spectral_feb2023_fix/center/K_alpha_center_fix/rec_k_alpha_center{i:03}.tif'
    beta_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/rec_spectral_feb2023_fix/center/K_beta_center_fix/rec_k_beta_center{i:03}.tif'

    im_poly[i, :, :] = io.imread(poly_path)
    im_alpha[i, :, :] = io.imread(alpha_path)
    im_beta[i, :, :] = io.imread(beta_path)

print(f'im_poly {im_poly.shape}')
print(f'im_alpha {im_alpha.shape}')
print(f'im_beta {im_beta.shape}')


# %%
show_hor_slices(im_poly)
show_vert_slices(im_poly)

# %%
sigma = 2

im_alpha_gauss_filtered = filter_im_array(im_alpha, sigma=sigma)
im_beta_gauss_filtered = filter_im_array(im_beta, sigma=sigma)
im_poly_gauss_filtered = filter_im_array(im_poly, sigma=sigma)

show_hor_slices(im_poly_gauss_filtered)
show_vert_slices(im_poly_gauss_filtered)

# %%
thresh = filters.threshold_minimum(im_poly[50])
thresh = 0.25
print(thresh)

mask = im_poly > thresh

show_hor_slices(im_poly)
show_hor_slices(mask)
show_vert_slices(im_poly)
show_vert_slices(mask)

# %%
show_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim)

# %%
show_full_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim)

# %%
show_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim, mask)

# %%
show_full_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim, mask)

# %%
z_size, y_size, x_size = 199, 150, 150
im_poly = np.empty((z_size, y_size, x_size))
im_alpha = np.empty((z_size, y_size, x_size))
im_beta = np.empty((z_size, y_size, x_size))


for i in np.arange(199):

    poly_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/rec_spectral_feb2023_fix/edges/Poly_edges_fix/rec_poly_edges{i:03}.tif'
    alpha_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/rec_spectral_feb2023_fix/edges/K_alpha_edges_fix/rec_k_alpha_edges{i:03}.tif'
    beta_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/rec_spectral_feb2023_fix/edges/K_beta_edges_fix/rec_k_beta_edges{i:03}.tif'

    im_poly[i, :, :] = io.imread(poly_path)
    im_alpha[i, :, :] = io.imread(alpha_path)
    im_beta[i, :, :] = io.imread(beta_path)

print(f'im_poly {im_poly.shape}')
print(f'im_alpha {im_alpha.shape}')
print(f'im_beta {im_beta.shape}')


# %%
show_hor_slices(im_poly)
show_vert_slices(im_poly)

# %%
sigma = 2

im_alpha_gauss_filtered = filter_im_array(im_alpha, sigma=sigma)
im_beta_gauss_filtered = filter_im_array(im_beta, sigma=sigma)
im_poly_gauss_filtered = filter_im_array(im_poly, sigma=sigma)

show_hor_slices(im_poly_gauss_filtered)
show_vert_slices(im_poly_gauss_filtered)

# %%
thresh = filters.threshold_minimum(im_poly[50])
thresh = 0.25
print(thresh)

mask = im_poly > thresh

show_hor_slices(im_poly)
show_hor_slices(mask)
show_vert_slices(im_poly)
show_vert_slices(mask)

# %%
show_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim)

# %%
show_full_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim)

# %%
show_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim, mask)

# %%
show_full_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, xlim, ylim, mask)

# %%
slice_number = slice(100, 149)

xlim = [np.min(im_alpha_gauss_filtered), np.max(im_alpha_gauss_filtered)]
ylim = [np.min(im_beta_gauss_filtered), np.max(im_beta_gauss_filtered)]

fig = plt.figure(figsize=(5, 5))

hist2d_data = plt.hist2d(
    im_alpha_gauss_filtered[slice_number][mask[slice_number]].flatten(), 
    im_beta_gauss_filtered[slice_number][mask[slice_number]].flatten(), 
    bins=64, 
    norm = LogNorm(),
)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

plt.imshow(hist2d_data[0].T, norm = LogNorm(), origin='lower')
plt.xticks(hist2d_data[1])
plt.yticks(hist2d_data[2])
plt.show()

# %%
