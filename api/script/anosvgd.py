import numpy as np
from sklearn.preprocessing import StandardScaler
import json


class AnoSVGD:
    def __init__(self, n_particles=100, lr=0.01, n_iter=1000, kernel_bandwidth=1.0):
        self.n_particles = n_particles
        self.lr = lr
        self.n_iter = n_iter
        self.kernel_bandwidth = kernel_bandwidth
        self.particles = None
        self.mean = None
        self.cov_inv = None

    def rbf_kernel(self, X, Y):
        pairwise_dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        return np.exp(-pairwise_dists / (2 * self.kernel_bandwidth ** 2))

    def grad_log_p_gaussian(self, particles):
        return -np.matmul(particles - self.mean, self.cov_inv)

    def svgd_update(self, particles, grad_log_p):
        for _ in range(self.n_iter):
            K = self.rbf_kernel(particles, particles)
            grad_K = -np.matmul(K, particles) + np.sum(K, axis=1, keepdims=True) * particles
            grad_log_p_particles = grad_log_p(particles)
            phi = (np.matmul(K, grad_log_p_particles) + grad_K) / particles.shape[0]
            particles += self.lr * phi
        return particles

    def fit(self, data):
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # 初始化粒子
        self.particles = np.random.randn(self.n_particles, data_scaled.shape[1])

        # 定义高斯分布的参数
        self.mean = np.zeros(data_scaled.shape[1])
        cov = np.eye(data_scaled.shape[1])
        self.cov_inv = np.linalg.inv(cov)

        # 定义目标分布的对数梯度
        grad_log_p = lambda x: self.grad_log_p_gaussian(x)

        # 运行SVGD
        self.particles = self.svgd_update(self.particles, grad_log_p)

    def detect(self, data, threshold_percentile=95):
        #
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        distances = np.linalg.norm(self.particles[:, None, :] - data_scaled[None, :, :], axis=-1)
        anomaly_scores = np.min(distances, axis=0)
        # 设定阈值
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        # 检测异常值
        anomalies = data[anomaly_scores > threshold].flatten()
        return anomalies

    def outlier_level(self, value, mean_temp, std_temp):
        """
        异常值也要分层, 如果大于4Sigma，则为比较异常，3.5Sigma到4Sigma之间,则为中等异常, 3Sigma到3.5Sigma则为轻度异常
        """
        outlier_level_score = 0.1
        outlier_level_describe = "需要关注"
        four_lower_bound = mean_temp - 4.0 * std_temp
        four_upper_bound = mean_temp + 4.0 * std_temp
        middle_lower_bound = mean_temp - 3.5 * std_temp
        middle_upper_bound = mean_temp + 3.5 * std_temp
        three_lower_bound = mean_temp - 3.0 * std_temp
        three_upper_bound = mean_temp + 3.0 * std_temp
        if value < four_lower_bound or value > four_upper_bound:
            outlier_level_score = 1.0
            outlier_level_describe = "严重异常"
        else:
            if value < middle_lower_bound or value > middle_upper_bound:
                outlier_level_score = 0.5
                outlier_level_describe = "中度异常"
            else:
                if value < three_lower_bound or value > three_upper_bound:
                    outlier_level_score = 0.2
                    outlier_level_describe = "轻度异常"
        return outlier_level_score, outlier_level_describe


# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    # data = np.random.randn(100, 3)  # 100个样本，每个样本有3个特征
    normal_data = np.random.normal(loc=1000, scale=100, size=300)  # 正常数据
    anomalous_data = np.random.normal(loc=200, scale=100, size=1)  # 异常数据
    input_data = np.concatenate([normal_data, anomalous_data]).reshape(-1, 1)
    print(f'input:{json.dumps(input_data.tolist())}')
    # 创建AnoSVGD实例
    anosvgd = AnoSVGD(n_particles=100, lr=0.01, n_iter=100, kernel_bandwidth=1.0)
    # 拟合数据
    anosvgd.fit(input_data)
    # 检测异常
    anomalies = anosvgd.detect(input_data, threshold_percentile=95)
    print(f'异常值为:' + str(anomalies))
