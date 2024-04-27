import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from json import dumps
from numpy import log10, nan
from itertools import product

def read_data(infile, outfile):
    # read in the data from the excel file, remove values of 1. log10 transform parasite density
    density = pd.read_excel(infile, header=(0, 1))
    density = density.set_index(('Unnamed: 0_level_0', 'day post-challenge'))
    density = density.replace(1.0, nan)
    density = density.mask((density < 20))
    density = log10(density)

    # get unqiue challenges and volunteer ids
    challenges = density.columns.get_level_values(0).unique()
    ids = density.columns.get_level_values(1).unique()
    # print(challenges)
    # print(ids)

    # change index name to "day" and set as a column
    density.index.names = ['day']
    density = density.reset_index()

    # a dummy empty dataframe to use when a volunteer has a missing challenge
    empty = pd.DataFrame({'x':[], 'y':[]})

    # loop through all volunteer-challenge combinations (even missing ones)
    # centre the day and log10-parasite densities so that only the slope need be estimated
    # the intercept is not estimated 
    S = []
    for i in product(ids, challenges):
        try:
            Y = density[[('day', ''), (i[1], i[0])]].dropna()
            Y = Y - Y.mean()
            Y.columns = ['x', 'y']
            Y['id'] = i[0]
            Y['challenge'] = i[1]
        except KeyError:
            Y = empty
        S.append(Y)

    # concatentate all the data into vectors as this is the most efficient way to represent it in stan
    Y = pd.concat(S)

    # nobs for each volunteer-challenge
    nobs = [len(i) for i in S]

    # print(nobs)
    # print(len(Y), sum(nobs))

    # data structure for stan model
    data = {'N':len(ids), 'K':len(Y), 'x':list(Y['x'].values), 'y':list(Y['y'].values), 'nobs':nobs}

    # output data structure as json for stan to read in
    with open(outfile, 'w') as f:
        f.write(dumps(data, indent=2))
    return Y

infile = 'pmr_BIO-004.xlsx'
outfile = 'pmr_BIO-004.data.json'
Y = read_data(infile, outfile)

model = CmdStanModel(stan_file='pmr_BIO-004.stan', cpp_options={'STAN_THREADS':'false'})
fit = model.sample(data=outfile, show_console=False, show_progress=False, chains=1, parallel_chains=1)

# sns.lmplot(data=Y, x='x', y='y', hue='challenge')
# plt.show()

print(fit.summary())
sns.displot(fit.draws_pd()['DbetaII'])
plt.show()
