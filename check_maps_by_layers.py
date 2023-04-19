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
# z_size, y_size, x_size = 199, 153, 153
z_size, y_size, x_size = 300, 150, 150
im_poly = np.empty((z_size, y_size, x_size))
im_alpha = np.empty((z_size, y_size, x_size))
im_beta = np.empty((z_size, y_size, x_size))

for i in np.arange(z_size):

    spec_tomo_path = '/Users/grimax/Documents/Science/xtomo/spectroTomo/'
    poly_path = f'{spec_tomo_path}rec_spectral_bragg_mar2023/2023_03_27-29_res/log_poly_fix/rec_poly_{i:03}.tif'
    alpha_path = f'{spec_tomo_path}rec_spectral_bragg_mar2023/2023_03_27-29_res/log_k-alpha_fix/rec_alpha_{i:03}.tif'
    beta_path = f'{spec_tomo_path}rec_spectral_bragg_mar2023/2023_03_27-29_res/log_k-beta_fix/rec_beta_{i:03}.tif'

    # poly_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/Spectral_tomo_data/Poly_correct/tomo_poly{i:03}.tif'
    # alpha_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/Spectral_tomo_data/K-alpha_correct/tomo_k_alpha{i:03}.tif'
    # beta_path = f'/Users/grimax/Documents/Science/xtomo/spectroTomo/Spectral_tomo_data/K-beta_correct/tomo_k_beta{i:03}.tif'

    im_poly[i, :, :] = io.imread(poly_path)
    im_alpha[i, :, :] = io.imread(alpha_path)
    im_beta[i, :, :] = io.imread(beta_path)

print(f'im_poly {im_poly.shape}')
print(f'im_alpha {im_alpha.shape}')
print(f'im_beta {im_beta.shape}')

# %%
slices_y = slice(30, 110)
slices_x = slice(30, 110)
slices_z = slice(50, 300)

im_poly = im_poly[slices_z, slices_y, slices_x]
im_alpha = im_alpha[slices_z, slices_y, slices_x]
im_beta = im_beta[slices_z, slices_y, slices_x]

print(f'im_poly sliced {im_poly.shape}')
print(f'im_alpha sliced {im_alpha.shape}')
print(f'im_beta sliced {im_beta.shape}')


# %%
def show_hor_slices(im_array):
    vmin = np.min(im_array)
    vmax = np.max(im_array)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(ax):
        # index = 50 if i == 0 else 100 if i == 1 else 150 
        index = 100 if i == 0 else 175 if i == 1 else 225 
        im = axis.imshow(im_array[index, :, :], vmin=vmin, vmax=vmax)
        # im = axis.imshow(im_array[index, :, :])
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
            # im = axis.imshow(im_array[z_size//2, :, :])
        elif i == 1:
            im = axis.imshow(im_array[:, y_size//2, :], vmin=vmin, vmax=vmax)
            # im = axis.imshow(im_array[:, y_size//2, :])
        elif i == 2:
            im = axis.imshow(im_array[:, :, x_size//2], vmin=vmin, vmax=vmax)
            # im = axis.imshow(im_array[:, :, x_size//2])
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

# show_hor_slices(im_poly_gauss_filtered)
# show_vert_slices(im_poly_gauss_filtered)

# show_hor_slices(im_beta_gauss_filtered)
# show_vert_slices(im_beta_gauss_filtered)

show_hor_slices(im_alpha_gauss_filtered)
show_vert_slices(im_alpha_gauss_filtered)

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
thresh = filters.threshold_minimum(im_poly[225])
print('threshold_minimum', thresh)

mask = np.full(im_poly_gauss_filtered.shape, False)

thresh = 0.15
print('pores/begenat thresh', thresh)

mask[150:] = im_poly_gauss_filtered[150:] > thresh

thresh = 0.15
print('pores/NaCl thresh', thresh)

mask[:150] = im_poly_gauss_filtered[:150] > thresh

show_hor_slices(im_poly)
show_hor_slices(mask)
show_vert_slices(im_poly)
show_vert_slices(mask)

# %%
mask_pores = ~mask

show_hor_slices(im_poly)
show_hor_slices(mask_pores)
show_vert_slices(im_poly)
show_vert_slices(mask_pores)

# %%
thresh_begenat = 0.65
print(thresh_begenat)

mask_begenat = (im_poly_gauss_filtered < thresh_begenat) & ~mask_pores

show_hor_slices(im_poly)
show_hor_slices(mask_begenat)
show_vert_slices(im_poly)
show_vert_slices(mask_begenat)

# %%
thresh_NaCl = 2
print(thresh_NaCl)

mask_NaCl = (im_poly_gauss_filtered < thresh_NaCl) & ~mask_pores & ~mask_begenat

show_hor_slices(im_poly)
show_hor_slices(mask_NaCl)
show_vert_slices(im_poly)
show_vert_slices(mask_NaCl)

# %%
thresh_LiNbO3 = 5
print(thresh_LiNbO3)

mask_LiNbO3 = (im_poly_gauss_filtered < thresh_LiNbO3) & ~mask_pores & ~mask_begenat & ~mask_NaCl

show_hor_slices(im_poly)
show_hor_slices(mask_LiNbO3)
show_vert_slices(im_poly)
show_vert_slices(mask_LiNbO3)

# %%
print(f'mask_pores count {np.sum(mask_pores)}')
print(f'mask_begenat count {np.sum(mask_begenat)}')
print(f'mask_NaCl count {np.sum(mask_NaCl)}')
print(f'mask_LiNbO3 count {np.sum(mask_LiNbO3)}')


# %%
def bins_for_hist(im, bins=128):
    bin_min = im.min()
    bin_max = im.max()
    bin_step = (bin_max - bin_min) / bins
    return np.arange(bin_min, bin_max + bin_step, bin_step)


def plot_hist(im_a, im_b, im_p, slice_number=None, mask=None, bins=128, log=True):
    if slice_number is None:
        slice_number = slice(0, im_a.shape[0])
    if mask is None:
        mask = np.full(im_a.shape, True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(im_a[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
    ax[0].grid(True)
    ax[1].hist(im_b[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
    ax[1].grid(True)
    ax[2].hist(im_p[slice_number][mask[slice_number]].flatten(), bins=bins, log=log)
    ax[2].grid(True)
    plt.show()

    
plot_hist(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered)
plot_hist(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_pores)
plot_hist(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_begenat)
plot_hist(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_NaCl)
plot_hist(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_LiNbO3)


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

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist2d(image1, image2, bins=64, norm = LogNorm())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if x_label:
        ax.set_xlabel(x_label, fontsize=18)
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)

    plt.show()
    

def show_all_maps(
    im_a, im_b, im_p, 
    a_lim=None, b_lim=None, p_lim=None, 
    mask=None,
    slice_number=None, 
    bins=128,
    figsize=(30, 10),
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
    
    h, xedges, yedges, _ = ax[0].hist2d(_im_a, _im_b, bins=bins, norm = LogNorm())
    ab_max_index = np.unravel_index(np.argmax(h), h.shape)
    print('ab max index', xedges[ab_max_index[0]], yedges[ab_max_index[1]])
    # ax[0].axline((0, 0), (xedges[ab_max_index[0]], yedges[ab_max_index[1]]), linewidth=2, color='r')
    # ax[0].hist2d(_im_a, _im_b, bins=bins)
    ax[0].set_xlim(a_lim)
    ax[0].set_ylim(b_lim)
    ax[0].tick_params(axis='x', labelsize=22)
    ax[0].tick_params(axis='y', labelsize=22)    
    ax[0].set_xlabel('MoK⍺', fontsize=36)
    ax[0].set_ylabel('MoKβ', fontsize=36)

    h, xedges, yedges, _ = ax[1].hist2d(_im_a, _im_p, bins=bins, norm = LogNorm())
    ap_max_index = np.unravel_index(np.argmax(h), h.shape)
    print('ap max index', xedges[ap_max_index[0]], yedges[ap_max_index[1]])
    # ax[1].hist2d(_im_a, _im_p, bins=bins)
    ax[1].set_xlim(a_lim)
    ax[1].set_ylim(p_lim)
    ax[1].tick_params(axis='x', labelsize=22)
    ax[1].tick_params(axis='y', labelsize=22)    
    ax[1].set_xlabel('MoK⍺', fontsize=36)
    ax[1].set_ylabel('Ploy', fontsize=36)

    h, xedges, yedges, _ = ax[2].hist2d(_im_b, _im_p, bins=bins, norm = LogNorm())
    bp_max_index = np.unravel_index(np.argmax(h), h.shape)
    print('bp max index', xedges[bp_max_index[0]], yedges[bp_max_index[1]])
    # ax[2].hist2d(_im_b, _im_p, bins=bins)
    ax[2].set_xlim(b_lim)
    ax[2].set_ylim(p_lim)
    ax[2].tick_params(axis='x', labelsize=22)
    ax[2].tick_params(axis='y', labelsize=22)    
    ax[2].set_xlabel('MoKβ', fontsize=36)
    ax[2].set_ylabel('Ploy', fontsize=36)
    
    # plt.show()
    return ax



# %%
ax = show_all_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered)

ax[0].axline((0, 0), (0.194, 0.147), linewidth=2, color='r')
ax[1].axline((0, 0), (0.194, 0.285), linewidth=2, color='r')
ax[2].axline((0, 0), (0.147, 0.285), linewidth=2, color='r')

ax[0].axline((0, 0), (0.885, 0.740), linewidth=2, color='b')
ax[1].axline((0, 0), (0.885, 1.045), linewidth=2, color='b')
ax[2].axline((0, 0), (0.740, 1.045), linewidth=2, color='b')

ax[0].axline((0, 0), (1.733, 4.587), linewidth=2, color='lightgrey')
ax[1].axline((0, 0), (1.733, 2.130), linewidth=2, color='lightgrey')
ax[2].axline((0, 0), (4.587, 2.130), linewidth=2, color='lightgrey')

# show_all_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_pores)
# show_all_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_begenat)
# show_all_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_NaCl)
# show_all_maps(im_alpha_gauss_filtered, im_beta_gauss_filtered, im_poly_gauss_filtered, mask=mask_LiNbO3)

plt.show()

# %%
data = {
    'alpha': im_alpha_gauss_filtered.flatten(), 
    'beta': im_beta_gauss_filtered.flatten(), 
    'poly': im_poly_gauss_filtered.flatten(),
}
df = pd.DataFrame(data=data)
df.describe()

# sns.pairplot(df, kind="hist", plot_kws={'bins':64,'cbar':True}, diag_kws={'bins':64})

# %%
train_data = np.array(df)
means_init = [
    [0, 0, 0],
    [0.194, 0.147, 0.287],
    [0.875, 0.739, 1.082],
    [1.733, 4.586, 2.130],
]
clf = mixture.GaussianMixture(n_components=4, covariance_type='full', means_init=means_init)
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
rnd_indx = np.random.choice(range(im_alpha_gauss_filtered.size), size=10000, replace=False)
part_df = labeled_df.iloc[rnd_indx]
part_df.describe()

# %%
sns.pairplot(
    part_df, 
    diag_kind="hist", 
    hue='labels', 
    markers='.', 
    palette={0: 'blue', 1: 'red', 2: 'green', 3: 'lightgray'}, 
    diag_kws={'bins':64}
)

# %%
scaler = preprocessing.StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

print(df.describe())

scaled_df = pd.DataFrame(data=scaled_data, columns=['alpha filtered scaled', 'beta filtered scaled', 'poly filtered scaled'])
scaled_df.describe()

# %%
clustering = DBSCAN(eps=0.1, min_samples=20).fit(scaled_df)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
print(set(labels))

plt.hist(labels, bins=len(set(labels)))

# %%

# %%

# %%

# %%

# %%

# %%

# %%
show_sliced_map(
    im_alpha_gauss_filtered, 
    im_beta_gauss_filtered, 
    [-0.5, 4], [-1, 10], 
    mask=mask_pores,
    slice_number = slice(0, 300),
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
