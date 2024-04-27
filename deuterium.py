import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

rng = np.random.default_rng()

Thalf = 480
mu = np.log(2)/Thalf
T0 = 14
N = 10

p = lambda t: np.exp(-mu*(t-T0))
t = np.array([14, 21, 35, 60, 90] + list(range(120, 365, 30)))
n = len(t)

S = []
for i in range(N):
    M = rng.integers(100, 1000, n)
    R = rng.binomial(M, p(t))
    phat = R/M
    df = pd.DataFrame({'id':[i]*n, 'x':t, 'R':R, 'M':M, 'phat':phat})
    S.append(df)

# concatentate all the data into vectors as this is the most efficient way to represent it in stan
Y = pd.concat(S)

# sns.lineplot(data=Y, x='x', y='phat', hue='id')
# plt.show()
# exit()

# nobs for each volunteer-challenge
nobs = [len(i) for i in S]

print(nobs)
print(len(Y), sum(nobs))

# data structure for stan model
data = {'N':N, 'K':len(Y), 'T0':T0, 'x':list(Y['x'].values), 'R':list(Y['R'].values), 'M':list(Y['M'].values), 'nobs':nobs}

# output data structure as json for stan to read in
with open('deuterium.data.json', 'w') as f:
    f.write(json.dumps(data, indent=2, cls=NpEncoder))

model = CmdStanModel(stan_file='deuterium.stan', cpp_options={'STAN_THREADS':'false'})
fit = model.sample(data='deuterium.data.json', show_console=False, show_progress=False, chains=1, parallel_chains=1)

print(fit.summary())

sns.displot(fit.draws_pd()['mu'])
plt.show()

# plt.plot(t, Y)
# plt.show()