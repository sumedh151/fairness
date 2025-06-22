import numpy as np
import pandas as pd
from scipy.stats import random_correlation
import scipy.stats as stats
from tqdm import tqdm


class DataGenerator:
    """
    A class to generate a simulated dataset with both non-sensitive and sensitive variables.
    
    Attributes:
        n (int): Number of samples to generate.
        num_non_sensitive (int): Number of non-sensitive variables.
        num_sensitive (int): Number of sensitive variables.
        latent_dim (int, optional): Dimension of latent variables (not used in this implementation).
        non_sensitive_dist (str): Distribution for non-sensitive variables ('normal', 'exp', 'gamma', 'beta', 'lognorm').
        seed (int, optional): Seed for the random number generator to ensure reproducibility.
    
    Methods:
        generate_n_eigen_values(n):
            Generate eigenvalues for the correlation matrix.
        gen_mean_std():
            Generate mean and standard deviation for the normal distribution.
        correlation_to_covariance(correlation, std_devs):
            Convert correlation matrix to covariance matrix using standard deviations.
        generate_simulated_data():
            Generate the simulated dataset and return it as a Pandas DataFrame.
    """
    
    def __init__(self, n=10000, num_non_sensitive=4, num_sensitive=1, latent_dim=None, 
                 non_sensitive_dist='normal', seed=None):
        """
        Initialize the DataGenerator with parameters.
        
        Parameters:
            n (int): Number of samples.
            num_non_sensitive (int): Number of non-sensitive variables.
            num_sensitive (int): Number of sensitive variables.
            latent_dim (int, optional): Dimension of latent variables.
            non_sensitive_dist (str): Distribution for non-sensitive variables.
            seed (int, optional): Seed for random number generation.
        """
        self.n = n
        self.num_non_sensitive = num_non_sensitive
        self.num_sensitive = num_sensitive
        self.latent_dim = latent_dim
        self.non_sensitive_dist = non_sensitive_dist
        
        # Set random seed for reproducibility
        self.rng = np.random.default_rng(seed)
        
        total_vars = self.num_non_sensitive + self.num_sensitive + 1
        eigen_values = self.generate_n_eigen_values(total_vars)
        corrmat = random_correlation.rvs(eigen_values, random_state=self.rng)
        std_devs = self.rng.uniform(0.5, 2, size=total_vars)
        
        self.covmat = self.correlation_to_covariance(corrmat, std_devs)
        self.mvnorm = stats.multivariate_normal(mean=np.zeros(self.covmat.shape[0]), cov=self.covmat)

        non_sensitive_dists = []
        for i in range(self.num_non_sensitive):
            if self.non_sensitive_dist == 'normal':
                non_sensitive_dists.append(stats.norm(*self.gen_mean_std()))
            elif self.non_sensitive_dist == 'exp':
                scale = 1  
                non_sensitive_dists.append(stats.expon(scale=scale))
            elif self.non_sensitive_dist == 'gamma':
                shape = 2.0
                scale = 1.0
                non_sensitive_dists.append(stats.gamma(shape, scale=scale))
            elif self.non_sensitive_dist == 'beta':
                a = 2.0
                b = 5.0
                non_sensitive_dists.append(stats.beta(a, b))
            elif self.non_sensitive_dist == 'lognorm':
                shape = 0.954
                scale = 1.0
                non_sensitive_dists.append(stats.lognorm(s=shape, scale=scale))
            else:
                raise ValueError(f"Unrecognized distribution: {self.non_sensitive_dist}")
        self.non_sensitive_dists = non_sensitive_dists

        sensitive_dists = []
        for i in range(self.num_sensitive):
            sensitive_dists.append(stats.uniform())
        self.sensitive_dists = sensitive_dists
        
        self.distY = stats.uniform()
        
        tau_sensitive = []
        for i in range(self.num_sensitive):
            p = np.round(self.rng.uniform(0.1, 0.9), 1)
            tau_sensitive.append(1 - p)
        self.tau_sensitive = tau_sensitive

        self.p_y = np.round(self.rng.uniform(0.1, 0.9), 1)
        self.tau_y = 1 - self.p_y 

    def generate_n_eigen_values(self, n):
        """
        Generate eigenvalues for the correlation matrix.
        
        Parameters:
            n (int): Number of eigenvalues.
        
        Returns:
            numpy.ndarray: Generated eigenvalues.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        numbers = self.rng.random(n)
        numbers = numbers / np.sum(numbers) * (n)
        return numbers

    def gen_mean_std(self):
        """
        Generate mean and standard deviation for the normal distribution.
        
        Returns:
            tuple: Mean and standard deviation.
        """
        return np.round(self.rng.uniform(-1, 1), 2), np.round(self.rng.uniform(0, 1), 2)

    def correlation_to_covariance(self, correlation, std_devs):
        """
        Convert correlation matrix to covariance matrix using standard deviations.
        
        Parameters:
            correlation (numpy.ndarray): Correlation matrix.
            std_devs (numpy.ndarray): Standard deviations.
        
        Returns:
            numpy.ndarray: Covariance matrix.
        """
        cov = correlation * np.outer(std_devs, std_devs)
        return cov    

    def generate_simulated_data(self):
        """
        Generate the simulated dataset.
        
        Returns:
            pandas.DataFrame: Simulated dataset with non-sensitive, sensitive variables, and target variable.
        """
        total_independent = self.num_non_sensitive + self.num_sensitive
        sigma_11 = self.covmat[0:total_independent, 0:total_independent]
        sigma_12 = self.covmat[0:total_independent, total_independent:]
        sigma_22 = self.covmat[total_independent:, total_independent:]
        sigma_21 = self.covmat[total_independent:, 0:total_independent]
        z_y_var = sigma_22 - np.matmul(np.matmul(sigma_21, np.linalg.inv(sigma_11)), sigma_12).flatten()[0]
        
        z = self.mvnorm.rvs(self.n)
        u = stats.norm.cdf(z)
        
        X = [self.non_sensitive_dists[i].ppf(u[:, i]) for i in range(self.num_non_sensitive)]
        
        S = []
        for i in range(self.num_sensitive):
            sensitive_data = self.sensitive_dists[i].ppf(u[:, self.num_non_sensitive + i])
            sensitive_data_bin = np.zeros(len(sensitive_data), dtype=int)
            sensitive_data_bin[sensitive_data >= self.tau_sensitive[i]] = 1
            S.append(sensitive_data_bin)
        
        xY = self.distY.ppf(u[:, self.num_non_sensitive + self.num_sensitive])
        y = np.zeros(len(xY), dtype=int)
        y[xY >= self.tau_y] = 1
        
        y_bayes_est = np.zeros(y.shape)
        for j in range(len(y)):
            z_xab = z[j, :total_independent]
            z_y_mu = np.matmul(np.matmul(sigma_21, np.linalg.inv(sigma_11)), z_xab)
            z_y_sample = self.rng.normal(z_y_mu[0], np.sqrt(z_y_var[0]), size=10000)
            x7_sample = self.distY.ppf(stats.norm.cdf(z_y_sample)).flatten()
            y_sample = np.zeros(len(x7_sample), dtype=int)
            y_sample[x7_sample >= self.tau_y] = 1
            y_bayes_est[j] = y_sample.mean()
        
        return pd.DataFrame({f'X{x_i+1}': X[x_i] for x_i in range(len(X))} | {f'S{s_i+1}': S[s_i] for s_i in range(len(S))} | {'Y': y} | {'Y_bayes': y_bayes_est})

