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
import pandas as pd

# %%
df = pd.read_csv(
    '/Users/grimax/Documents/Science/xtomo/spectroTomo/Mo source spectra 2020/Mo_source_spectra.csv', sep=';', decimal=','
).drop(columns=['Unnamed: 0'])
df

# %%
# df['no filter'].plot(logy=True)
df.plot(x='keV', logy=True)

# %%
