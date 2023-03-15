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

# %%
na_cl_alpha = 1/0.6
na_cl_beta = 1/0.839
na_cl_poly = 0.6 # from recon images

li_nb_o3_alpha = 1/0.2
li_nb_o3_beta = 1/0.04
li_nb_o3_poly = 1.5 # from recon images ### 1.03 — calc for 24.8 keV

ag_c22_h43_o2_alpha = 1/0.151 # 2.5—3.0 from recon images
ag_c22_h43_o2_beta = 1/0.203 # 4.5—5.0 from recon images
ag_c22_h43_o2_poly = 2.6 # 3.5—4.0 from recon images ### 2.6 — calc for 24.8

print(f'na_cl_alpha\t{na_cl_alpha:.2f},\nna_cl_beta\t{na_cl_beta:.2f}\nna_cl_poly\t{na_cl_poly:.2f}\n')
print(f'li_nb_o3_alpha\t{li_nb_o3_alpha:.2f},\nli_nb_o3_beta\t{li_nb_o3_beta:.2f}\nli_nb_o3_poly\t{li_nb_o3_poly:.2f}\n')
print(
    f'ag_c22_h43_o2_alpha\t{ag_c22_h43_o2_alpha:.2f},\n' +
    f'ag_c22_h43_o2_beta\t{ag_c22_h43_o2_beta:.2f}\n' + 
    f'ag_c22_h43_o2_poly\t{ag_c22_h43_o2_poly:.2f}\n'
)

# %%
1/0.077

# %%
# 1.67 — 24.8keV NaCl

# %%
1/0.670

# %%
# 1.5 — 24.8 LiNbO3

# %%
1/0.385

# %%
# 2.6 — 24.8 ag_c22_h43_o2

# %%
# data for centroids
na_cl_alpha = 1.1
na_cl_beta = 0.8
na_cl_poly = 0.3

li_nb_o3_alpha = 0.3
li_nb_o3_beta = 2.2
li_nb_o3_poly = 1

ag_c22_h43_o2_alpha = 3
ag_c22_h43_o2_beta = 4.5
ag_c22_h43_o2_poly = 2


# %%
# NaCl
#
# KeV/g/cm^3 2.165    1        0.5
# 17489.7    599.572  1298.07  2596.15
# 19593.7    836.786  1811.64  3623.28
# 24731.6    1651.17  3574.78  7149.55
#
#
# LiNbO3
#
# KeV/g/cm^3 4.5      2        1
# 17489.7    206.737  465.157  930.314
# 19593.7    42.6398  95.9396  191.879
# 24731.6    997.664  176.540  353.079
#
#
# AgC22H43O2
#
# KeV/g/cm^3 4        2        1
# 17489.7    397.368  794.735  1589.47
# 19593.7    539.929  1079.86  2159.72
# 24731.6    78.4620  1995.33  3990.66
# 26326.1    202.441  404.882  809.765
#

energy = np.array([17489.7, 19593.7, 24731.6, 26326.1])
na_cl_density = np.array([2.165, 1, 0.5])
na_cl = np.array([[599.572, 1298.07, 2596.15], [836.786, 1811.64, 3623.28], [1651.17, 3574.78, 7149.55]])
li_nb_o3_density = np.array([4.5, 2, 1])
li_nb_o3 = np.array([[206.737, 465.157, 930.314], [42.6398, 95.9396, 191.879], [997.664, 176.540, 353.079]])
ag_c22_h43_o2_density = np.array([4, 2, 1])
ag_c22_h43_o2 = np.array([[397.368, 794.735, 1589.47], [539.929, 1079.86, 2159.72], [78.4620, 1995.33, 3990.66], [202.441, 404.882, 809.765]])


# %%
print(1000/na_cl[:,0])
print(1000/li_nb_o3[:,2])
print(1000/ag_c22_h43_o2[:,2])

# %%
(0.25058512+1.23492618)/2

# %%
1000/1651.17

# %%
import numpy as np
li_nb_o3 = np.array([[206.737, 465.157, 930.314], [42.6398, 95.9396, 191.879], [997.664, 176.540, 353.079]])
li_nb_o3[:,2]

# %%
