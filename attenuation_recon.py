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
import numpy as np
from skimage import io, filters
import matplotlib.pyplot as plt

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


# %%
sigma = 2
im_alpha_gauss_filtered = filter_im_array(im_alpha, sigma=sigma)
im_beta_gauss_filtered = filter_im_array(im_beta, sigma=sigma)
im_poly_gauss_filtered = filter_im_array(im_poly, sigma=sigma)

# %%
thresh = filters.threshold_minimum(im_poly[50])
print(thresh)

mask = im_poly > thresh

# %%
show_hor_slices(im_alpha_gauss_filtered)
show_hor_slices(im_beta_gauss_filtered)
show_hor_slices(im_poly_gauss_filtered)
# show_hor_slices(im_poly_gauss_filtered)
# show_vert_slices(im_poly)
# show_vert_slices(im_poly_gauss_filtered)

# %%
slice_number = slice(145, 155)

plt.hist(im_poly[slice_number][mask[slice_number]].flatten(), bins=64, log=True)
plt.grid(True)
plt.show()

plt.hist(im_poly_gauss_filtered[slice_number][mask[slice_number]].flatten(), bins=64, log=True)
plt.grid(True)
plt.show()

# %%
plt.hist(im_alpha_gauss_filtered[mask].flatten(), bins=64, log=True)
plt.grid(True)
plt.show()

plt.hist(im_beta_gauss_filtered[mask].flatten(), bins=64, log=True)
plt.grid(True)
plt.show()

plt.hist(im_poly_gauss_filtered[mask].flatten(), bins=64, log=True)
plt.grid(True)
plt.show()

# %%
