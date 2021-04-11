#!/usr/bin/env python3
from graphTools import *
import sys

args = parserArguments(sys.argv)
df = lireDataFrame(args)

# see customizing-with-matplotlibrc-files
# https: // matplotlib.org/tutorials/introductory/customizing.html

sns.set(style="darkgrid", rc={'text.usetex': False,
                              'legend.handletextpad': 0,
                              'figure.titlesize': 'medium'})

# plt.style.use(['dark_background'])


# Selection des lignes :
# df = df[(-df.threads.isin([8])) & (df.kernel.isin(['mandel']))].reset_index(drop = True)

# Creation du graphe :
fig = creerGraphique(df=df, args=args)
fig.tight_layout()
engeristrerGraphique(fig)
