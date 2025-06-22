import numpy as np
import pandas as pd
from scipy.stats import random_correlation
import scipy.stats as stats
from tqdm import tqdm

def generate_n_eigen_values(n):
    if n <= 0:
        raise ValueError("n must be a positive integer")
    numbers = np.random.random(n)    
    numbers = numbers / np.sum(numbers) * (n)
    return numbers

def gen_mean_std():
    return np.round(np.random.uniform(-1, 1), 2) , np.round(np.random.uniform(0, 1), 2)

def correlation_to_covariance(correlation, std_devs):
    cov = correlation * np.outer(std_devs, std_devs)
    return cov

def generate_simulated_data(n=10000, n_datasets=100, num_non_sensitive=4, num_sensitive=2, latent_dim=None, non_sensitive_dist='normal'):
    """
    Generate a list of simulated datasets with both non-sensitive and sensitive variables.

    Parameters:
    - n (int): The number of samples in each dataset. Default is 10,000.
    - n_datasets (int): The number of datasets to sample. Default is 100.
    - num_non_sensitive (int): The number of non-sensitive variables in each dataset. Default is 4.
    - num_sensitive (int): The number of sensitive variables in each dataset. Default is 2.
    - non_sensitive_dist (str): Distribution type for non-sensitive variables. Options are:
      - 'normal': Normal distribution
      - 'exp': Exponential distribution
      - 'gamma': Gamma distribution
      - 'beta': Beta distribution
      - 'lognorm': Log-normal distribution
      Default is 'normal'.

    Returns:
    - List[pd.DataFrame]: A list containing `n_datasets` DataFrames, each representing a simulated dataset with the following columns:
      - Non-sensitive variables: `X1`, `X2`, ..., up to `X{num_non_sensitive}`
      - Sensitive variables: `S1`, `S2`, ..., up to `S{num_sensitive}`
      - Binary outcome variable: `Y`

    Raises:
    - ValueError: If an unrecognized distribution is provided for `non_sensitive_dist`.

    Notes:
    - The function generates a multivariate normal distribution for the latent variables and transforms them to produce the non-sensitive and sensitive variables.
    - Thresholds for sensitive variables and the binary outcome variable are determined based on uniform distributions and applied to the generated data.

    Example:
    >>> datasets = generate_simulated_data(n=5000, n_datasets=10, num_non_sensitive=3, num_sensitive=2, non_sensitive_dist='gamma')
    >>> len(datasets)
    10
    >>> datasets[0].head()
         X1        X2        X3  S1  S2  Y
    0  ...       ...       ...   0   1  0
    1  ...       ...       ...   1   0  1
    """
    total_vars = num_non_sensitive + num_sensitive + 1
    eigen_values = generate_n_eigen_values(total_vars)
    rng = np.random.default_rng()
    corrmat = random_correlation.rvs(eigen_values, random_state=rng)
    std_devs = np.random.uniform(0.5, 2, size=total_vars)
    covmat = correlation_to_covariance(corrmat , std_devs)
    mvnorm = stats.multivariate_normal(mean=np.zeros(covmat.shape[0]), cov=covmat) 

    total_independent = num_non_sensitive + num_sensitive
    sigma_11=covmat[0:total_independent, 0:total_independent]
    sigma_12=covmat[0:total_independent, total_independent:]
    sigma_22=covmat[total_independent:,  total_independent:]
    sigma_21=covmat[total_independent:,  0:total_independent]
    z_y_var = sigma_22-np.matmul(np.matmul(sigma_21,np.linalg.inv(sigma_11)),sigma_12).flatten()[0] 

    non_sensitive_dists = []
    for i in range(num_non_sensitive):
        if non_sensitive_dist == 'normal':
            non_sensitive_dists.append(stats.norm(*gen_mean_std()))
        elif non_sensitive_dist == 'exp':
            scale = 1  
            non_sensitive_dists.append(stats.expon(scale=scale))
        elif non_sensitive_dist == 'gamma':
            shape = 2.0
            scale = 1.0
            non_sensitive_dists.append(stats.gamma(shape, scale=scale))
        elif non_sensitive_dist == 'beta':
            a = 2.0
            b = 5.0
            non_sensitive_dists.append(stats.beta(a,b))
        elif non_sensitive_dist == 'lognorm':
            shape = 0.954
            scale = 1.0
            non_sensitive_dists.append(stats.lognorm(s=shape, scale=scale))
        else:
            raise ValueError(f"Unrecognized distribution: {non_sensitive_dist}")

    sensitive_dists = []
    for i in range(num_sensitive):
        sensitive_dists.append(stats.uniform())

    distY = stats.uniform()
    
    tau_sensitive = []
    for i in range(num_sensitive):
        p = np.round(np.random.uniform(0.1, 0.9), 1)
        tau_sensitive.append(1 - p)

    p_y = np.round(np.random.uniform(0.1, 0.9), 1)
    tau_y = 1 - p_y 
    
    datasets = []
    
    for n_i in tqdm(range(n_datasets)):
        z = mvnorm.rvs(n)
        u = stats.norm.cdf(z)
        
        X = []
        for i in range(num_non_sensitive):
            X.append(non_sensitive_dists[i].ppf(u[:, i]))
            
        S = []
        for i in range(num_sensitive):
            sensitive_data = sensitive_dists[i].ppf(u[:, num_non_sensitive + i])
            sensitive_data_bin = np.zeros(len(sensitive_data), dtype=int)
            sensitive_data_bin[sensitive_data >= tau_sensitive[i]] = 1
            S.append(sensitive_data_bin)
        
        xY = distY.ppf(u[:, num_non_sensitive + num_sensitive])
        y = np.zeros(len(xY), dtype=int)
        y[xY >= tau_y] = 1     
                
        y_bayes_est=np.zeros(y.shape)
        for j in range(len(y)):
            z_xab = z[j,:total_independent]
            z_y_mu = np.matmul(np.matmul(sigma_21,np.linalg.inv(sigma_11)),z_xab)
            z_y_sample = np.random.normal(z_y_mu[0],np.sqrt(z_y_var[0]),size=10000)
            x7_sample =  distY.ppf(stats.norm.cdf(z_y_sample)).flatten()
            y_sample = np.zeros(len(x7_sample), dtype=int)
            y_sample[x7_sample >= tau_y] = 1
        #     x7_cond_exp = m7.ppf(stats.norm.cdf(z_y_mu)).flatten()[0]
            y_bayes_est[j] = y_sample.mean()#1 if x7_cond_exp>tau_y else 0
        
        datasets.append(pd.DataFrame({f'X{x_i+1}': X[x_i] for x_i in range(len(X))} | {f'S{s_i+1}': S[s_i] for s_i in range(len(S))} | {'Y':y} | {'Y_bayes':y_bayes_est}))

    return datasets