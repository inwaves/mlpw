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



