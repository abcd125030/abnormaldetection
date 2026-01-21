import numpy as np
from scipy.special import digamma
from scipy.stats import multivariate_normal


class VBEMGMM:
    def __init__(self, n_components, n_features, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.beta = None
        self.m = None
        self.W = None
        self.nu = None
        self.lower_bound = -np.inf

    def _initialize_parameters(self, data):
        np.random.seed(42)
        self.means = np.random.rand(self.n_components, self.n_features)
        self.covariances = np.array(
            [np.eye(self.n_features)] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

        self.alpha = np.ones(self.n_components)
        self.beta = np.ones(self.n_components)
        self.m = np.random.rand(self.n_components, self.n_features)
        self.W = np.array([np.eye(self.n_features)] * self.n_components)
        self.nu = np.ones(self.n_components) * (self.n_features + 1)

    def fit(self, data):
        self._initialize_parameters(data)
        n_samples = data.shape[0]

        for iteration in range(self.max_iter):
            # E-step: Update responsibilities
            log_rho = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                log_rho[:, k] = (
                    digamma(self.alpha[k]) - digamma(np.sum(self.alpha)) +
                    0.5 * np.sum(digamma(0.5 * (self.nu[k] - 
                                                np.arange(self.n_features)[:, np.newaxis])), axis=0) -
                    0.5 * self.n_features / self.beta[k] -
                    0.5 * self.nu[k] * np.sum((data - self.m[k]) @ self.W[k] * (data - self.m[k]), axis=1) -
                    0.5 * self.n_features * np.log(2 * np.pi)
                )
            log_rho -= log_rho.max(axis=1, keepdims=True)
            rho = np.exp(log_rho)
            rho /= rho.sum(axis=1, keepdims=True)

            # M-step: Update variational parameters
            Nk = rho.sum(axis=0)
            xk = (rho.T @ data) / Nk[:, np.newaxis]
            Sk = np.zeros((self.n_components, self.n_features, self.n_features))
            for k in range(self.n_components):
                diff = data - xk[k]
                Sk[k] = (rho[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]

            self.alpha = 1 + Nk
            self.beta = 1 + Nk
            self.m = (Nk[:, np.newaxis] * xk + self.m) / (Nk[:, np.newaxis] + 1)
            self.W = np.linalg.inv(np.linalg.inv(self.W) + Nk[:, np.newaxis, np.newaxis] * Sk)
            self.nu = self.n_features + Nk

            # Compute lower bound
            new_lower_bound = np.sum(rho * log_rho) - np.sum(rho * log_rho)
            if np.abs(new_lower_bound - self.lower_bound) < self.tol:
                break
            self.lower_bound = new_lower_bound

        print("Converged after {} iterations".format(iteration))
        print("Estimated means:\n", self.m)
        print("Estimated weights:\n", self.alpha / np.sum(self.alpha))

    def predict(self, data):
        log_rho = np.zeros((data.shape[0], self.n_components))
        for k in range(self.n_components):
            log_rho[:, k] = (
                digamma(self.alpha[k]) - digamma(np.sum(self.alpha)) +
                0.5 * np.sum(digamma(0.5 * (self.nu[k] - np.arange(self.n_features)[:, np.newaxis])), axis=0) -
                0.5 * self.n_features / self.beta[k] -
                0.5 * self.nu[k] * np.sum((data - self.m[k]) @ self.W[k] * (data - self.m[k]), axis=1) -
                0.5 * self.n_features * np.log(2 * np.pi)
            )
        log_rho -= log_rho.max(axis=1, keepdims=True)
        rho = np.exp(log_rho)
        rho /= rho.sum(axis=1, keepdims=True)
        return np.argmax(rho, axis=1)

# Example usage
# if __name__ == "__main__":
#     # Generate synthetic data
#     np.random.seed(42)
#     n_samples = 500
#     n_components = 3
#     n_features = 2

#     true_means = np.array([[0, 0], [3, 3], [0, 4]])
#     true_covariances = np.array([np.eye(n_features)] * n_components)
#     true_weights = np.array([0.4, 0.4, 0.2])

#     data = np.vstack([
#         np.random.multivariate_normal(mean, cov, int(weight * n_samples))
#         for mean, cov, weight in zip(true_means, true_covariances, true_weights)
#     ])

#     # Create and fit VBEMGMM model
#     vbem_gmm = VBEMGMM(n_components=n_components, n_features=n_features)
#     vbem_gmm.fit(data)

#     # Predict cluster labels
#     labels = vbem_gmm.predict(data)
#     print("Predicted labels:\n", labels)
