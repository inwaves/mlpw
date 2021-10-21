import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def rbf_kernel(x1, x2, var_sigma, length_scale):
    d = cdist(x1, x2) if x2 is not None else cdist(x1, x1)
    K = var_sigma * np.exp(-np.power(d, 2)/length_scale)
    
    return K


# Choose index set for the marginal.
index_set_size = 10 # The fewer points sampled, the less "smooth", more step-like the function is. I don't know the term here.
x = np.linspace(-6, 6, index_set_size).reshape(-1, 1) # Reshape puts each element in an array of its own.
print(x.shape)
# Compute covariance matrix by feeding in params to our covariance function.
var_sigma = 1000.0 # I can't tell what the effect of this is by eyeballing.
length_scale = 20.0 # Increasing this smooths out the function.
K = rbf_kernel(x, None, var_sigma, length_scale)

# Create mean vector
mu = np.zeros(x.shape)

# Draw samples from the Gaussian distribution.
sample_count = 10
f = np.random.multivariate_normal(mu, K, sample_count)

# Plot samples.
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, f.T);



