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
import matplotlib.pyplot as plt

# %%
U = 140 # KeV
K = 9.2e-7 # 1/kV
Z = 74 # W
h = 6.626e-34 # Plank const in Js

u = np.arange(0, U+1, 1)
v = u / h
Vmax = U / h

I = Z * h * (Vmax - v)
n = K*Z*u

plt.plot(I*n)

# %%
