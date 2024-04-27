/* 
Model for data in BIO-004. Priors are set from estimates of Sandoval data. 
*/
functions {
    tuple(int, int, int) next_volunteer(int j, array[] int nobs, array[] int pos) {
        // return indices of data vectors
        return (j, pos[j], pos[j] + nobs[j] - 1);
    }
}

data {
    int<lower=0> N;   // no. of individuals
    int<lower=0> K;   // total no. of observations
    vector[K] x;
    vector[K] y;
    array[3*N] int<lower=0> nobs; // observations for each individual-challenge combination
}

transformed data {
    array[3*N] int pos;
    
    // cumulation position of observations
    pos[1] = 1;
    for (i in 2:3*N-1)
        pos[i] = pos[i-1] + nobs[i-1];
    pos[3*N] = K; // K not K+1 as slicing checks crash the code otherwise   
}

parameters {
    vector[N] beta; // slope of parasite density vs time in challenge I for each volunteer
    real DbetaII; // across-volunteer change in slope between challenge I and II
    real DbetaIII; // across-volunteer change in slope between challenge I and III
    real<lower=0> sigma; // st. dev. of error
    real<lower=0> beta_mu;
    real<lower=0> beta_sigma;
}

model {
    int j = 0;

    sigma ~ gamma(2, 5);
    DbetaII ~ normal(0, 0.1);
    DbetaIII ~ normal(0, 0.1);
    beta_mu ~ normal(0.63, 0.1);
    beta_sigma ~ gamma(2, 15);
    beta ~ normal(beta_mu, beta_sigma);

    // loop through volunteers
    for (i in 1:N) {
        int p, q;
        // challenge I
        (j, p, q) = next_volunteer(j+1, nobs, pos);
        y[p:q] ~ normal(beta[i]*x[p:q], sigma);

        // challenge II
        (j, p, q) = next_volunteer(j+1, nobs, pos);
        y[p:q] ~ normal((beta[i] + DbetaII)*x[p:q], sigma);

        // challenge III
        (j, p, q) = next_volunteer(j+1, nobs, pos);
        y[p:q] ~ normal((beta[i] + DbetaIII)*x[p:q], sigma);
    }
}