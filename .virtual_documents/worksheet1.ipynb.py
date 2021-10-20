import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


mu = 0.2
N = 1000

# Take N samples from a binomial distribution with mean mu.
X = np.random.binomial(1, mu, N)
mu_test = np.linspace(0, 1, 1000)


mu_test


def posterior(a, b, X):
    a_n = a + X.sum()
    b_n = b + (X.shape[0] - X.sum())
    
    return beta.pdf(mu_test, a_n, b_n)


a, b = 10, 38
prior_mu = beta.pdf(mu_test, a, b)


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

ax.plot(mu_test, prior_mu, 'g')
ax.fill_between(mu_test, prior_mu, color='green', alpha=0.3)

ax.set_xlabel('$\mu$')
ax.set_ylabel('$p(\mu|\mathbf{x})$')


index = np.random.permutation(X.shape[0])
for i in range(0, X.shape[0]):
    y = posterior(a, b, X[:index[i]])
    plt.plot(mu_test, y, 'r', alpha=0.3)
    plt.plot(X, y-mu_test)
    
y = posterior(a, b, X)
plt.plot(mu_test, y, 'b', linewidth=4.0)


plt.plot(X, y-mu_test)


import numpy as np
import matplotlib.pyplot as plt


def plot_line(ax, w):
    # Input data.
    X = np.zeros((2, 2))
    X[0, 0] = -5.0
    X[1, 0] = 5.0
    X[:, 1] = 1.0
    
    # Because of the concatenation we have to flip the transpose.
    y = w.dot(X.T)
    ax.plot(X[:, 0], y)


# Create prior distribution.
tau = 1.0 * np.eye(2)
w0 = np.zeros((2, 1))

w0


# Sample from prior.
n_samples = 100
w_sample = np.random.multivariate_normal(w0.flatten(), tau, size=n_samples)

w_sample[:5] # This is the prior p(w)


# Create plot.
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

for i in range(0, w_sample.shape[0]):
    plot_line(ax, w_sample[i, :])


"""
Create a contour plot of a two-dimensional normal distribution.

Parameters
----------
ax : axis handle to plot
mu : mean vector 2x1
Sigma : covariance matrix 2x2

"""
from scipy.stats import multivariate_normal

def plotdistribution(ax, mu, Sigma):
    x = np.linspace(-1.5, 1.5, 100) # Give me 100 equally-distributed numbers between [-1.5, 1.5]
    x1p, x2p = np.meshgrid(x, x) # Not sure!
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T 

    pdf = multivariate_normal(mu.flatten(), Sigma)
    Z = pdf.pdf(pos)
    Z = Z.reshape(100, 100)
    
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
    ax.contour(x1p, x2p, Z, 5, colors='r', lw=5, alpha=0.7)
    
    plt.show()


plotdistribution(ax, w_sample[:1], tau)


index = np.random.permutation(X.shape[0])
for i in range(0, index.shape[0]):
    X_i = X[index, :]
    y_i = y[index]
    
    # Compute posterior.
    
    # Visualise posterior.
    # Visualise samples from posterior with the data.
    # Print out the mean of the posterior.



