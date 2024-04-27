functions {
    tuple(int, int, int) next_volunteer(int j, array[] int indicies) {
        // get slice indicies in data vectors for next volunteer
        return (j, indicies[j], indicies[j+1] - 1);
    }
}

data {
    int<lower=0> N;             // no. of individuals
    int<lower=0> K;             // total no. of observations
    real T0;                    // time of first observation
    vector[K] x;                // time
    array[K] int<lower=0> R;    // no. of labelled cells observed
    array[K] int<lower=0> M;    // total no. of cells observed
    array[N] int<lower=0> nobs; // no. of observations for each individual
}

transformed data {
    array[N+1] int<lower=0> indicies;
   
    // first index of observations in data vectors for each volunteer
    // include N+1 to allow getting index of last datum of last volunteer
    indicies[1] = 1;
    for (i in 2:N+1)
        indicies[i] = indicies[i-1] + nobs[i-1];
}

parameters {
    vector<lower=0>[N] Thalf; // half-life for each volunteer
    real<lower=0> mu;    // mean hyperprior
    real<lower=0> sigma; // st. dev. hyperprior
}

model {
    vector[K] prop;

    // population proportions of labelled cells for each volunteer over time
    for (i in 1:N) {
        int p = indicies[i];
        int q = indicies[i+1]-1;
        prop[p:q] = exp(-log(2.0)/Thalf[i]*(x[p:q] - T0));
    }

    for (i in 1:N) {
        int p = indicies[i];
        int q = indicies[i+1]-1;
        R[p:q] ~ binomial(M[p:q], prop[p:q]);
    }
    
    // hyper-priors on population mean and st. dev. from Akondy 2017
    mu ~ normal(480, 20);
    sigma ~ gamma(2, 0.1);

    // half-life is normally distibuted in the population
    Thalf ~ normal(mu, sigma) T[0,];
}