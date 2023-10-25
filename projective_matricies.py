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
import json

# %%
with open('/Users/grimax/Documents/Science/xtomo/spectroTomo/Projective matricies/input_params_grid_center_with_pr_tr.json') as f:
    d = json.load(f)
    print(d)

# %%
k_alpha_pt = np.array(d['kAlpha']['PrTrCoeff'])
k_alpha_pti = np.array(d['kAlpha']['PrTrInverseCoeff'])

k_beta_pt = np.array(d['kBeta']['PrTrCoeff'])
print(k_alpha_pt)
print(k_beta_pt)

# %%
point_alpha = np.array([*d['kAlpha']['quad'][0:2], 1])
point_poly = np.array([*d['poly']['quad'][:2], 1])

print(point_alpha)
print(point_poly)

# %%
# result = k_alpha_pt @ point_poly
result = np.linalg.inv(k_alpha_pt) @ point_poly
result /= result[2]
print(result)


# %%
point_alpha = np.array([*d['kAlpha']['quad'][:2], 1])
point_poly = np.array([*d['poly']['quad'][:2], 1])

# %%

# %%

# %%

# %%
import numpy as np
import json

with open('input_params_grid_center_with_pr_tr.json') as f:
    d = json.load(f)
    
k_alpha_pt = np.array(d['kAlpha']['PrTrCoeff'])

point_alpha = np.array([*d['kAlpha']['quad'][:2], 1])
point_poly = np.array([*d['poly']['quad'][:2], 1])

result = k_alpha_pt @ point_poly
result /= result[-1]
print(f'{result} != {point_alpha}')

result = np.linalg.inv(k_alpha_pt) @ point_poly
result /= result[-1]
print(f'{result} == {point_alpha}')


# %%
