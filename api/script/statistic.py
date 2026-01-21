import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller


class OutlierDetectorThreeSigma:
    def __init__(self, data, threshold, over_3sigma_threshold_temp, lower_3sigma_threshold_temp):
        self.data = np.array(data)
        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        self.threshold = threshold
        self.over_3sigma_threshold_temp = over_3sigma_threshold_temp
        self.lower_3sigma_threshold_temp = lower_3sigma_threshold_temp
    
    def is_outlier(self, value):
        """
        判断一个值是否为异常值。
        如果值落在均值±3*标准差之外，则认为是异常值。
        """
        lower_bound = self.mean - self.threshold * self.std_dev
        upper_bound = self.mean + self.threshold * self.std_dev
        if self.over_3sigma_threshold_temp != -1.0:
            upper_bound = self.mean + self.over_3sigma_threshold_temp * self.std_dev
        if self.over_3sigma_threshold_temp != -1.0:
            upper_bound = self.mean + self.over_3sigma_threshold_temp * self.std_dev
        return value < lower_bound or value > upper_bound
    
    def find_outliers(self):
        """
        找出数据中的所有异常值。
        """
        return [value for value in self.data if self.is_outlier(value)]

    def get_up_down_level(self):
        """
        找出数据中的所有上下区间。
        """
        lower_bound = self.mean - self.threshold * self.std_dev
        if self.lower_3sigma_threshold_temp != -1.0:
            lower_bound = self.mean - self.lower_3sigma_threshold_temp * self.std_dev
        if lower_bound < 0.0:
            lower_bound = 0.0
        upper_bound = self.mean + self.threshold * self.std_dev
        if self.over_3sigma_threshold_temp != -1.0:
            upper_bound = self.mean + self.over_3sigma_threshold_temp * self.std_dev
        return round(lower_bound, 2), round(upper_bound, 2)

    def outlier_level(self, value, mean_temp, std_temp):
        """
        异常值也要分层, 如果大于4Sigma，则为比较异常，3.5Sigma到4Sigma之间,则为中等异常, 3Sigma到3.5Sigma则为轻度异常
        """
        outlier_level_score = 0.1
        outlier_level_describe = "需要关注"
        four_lower_bound = mean_temp - 4.0 * std_temp
        four_upper_bound = mean_temp + 4.0 * std_temp
        if self.over_3sigma_threshold_temp != -1.0:
            four_upper_bound = mean_temp + self.over_3sigma_threshold_temp * std_temp
        if self.lower_3sigma_threshold_temp != -1.0:
            four_lower_bound = mean_temp - self.lower_3sigma_threshold_temp * std_temp
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


class OutlierDetectorBoxPlot:
    def __init__(self):
        self.data = None
        self.anomalies = None
        self.threshold = 1.5

    def detect_anomalies(self, data, threshold=1.5):
        self.data = data
        self.threshold = threshold
        # 绘制箱线图
        # plt.boxplot(self.data)

        # 获取箱线图的特征
        quartiles = np.percentile(self.data, [25, 50, 75])
        iqr = quartiles[2] - quartiles[0]
        lower_bound = quartiles[0] - self.threshold * iqr
        upper_bound = quartiles[2] + self.threshold * iqr

        # 确定异常值
        self.anomalies = [x for x in self.data if x < lower_bound or x > upper_bound]

    def get_up_down_level(self):
        """
        找出数据中的所有上下区间。
        """
        # 获取箱线图的特征
        quartiles = np.percentile(self.data, [25, 50, 75])
        iqr = quartiles[2] - quartiles[0]
        lower_bound = quartiles[0] - self.threshold * iqr
        if lower_bound < 0.0:
            lower_bound = 0.0
        upper_bound = quartiles[2] + self.threshold * iqr
        return round(lower_bound, 2), round(upper_bound, 2)

    def get_anomalies(self):
        return self.anomalies

    def detect_anomaly_data(self, new_data):
        # 获取箱线图的特征
        quartiles = np.percentile(self.data, [25, 50, 75])
        iqr = quartiles[2] - quartiles[0]
        lower_bound = quartiles[0] - self.threshold * iqr
        upper_bound = quartiles[2] + self.threshold * iqr
        if new_data < lower_bound or new_data > upper_bound:
            return True
        return False

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


class OutlierDetectorARIMA:
    def __init__(self, order=(1, 1, 1), threshold=1.5):
        """
        初始化ARIMA异常检测器
        
        Parameters:
        order (tuple): ARIMA模型的(p, d, q)参数
        threshold (float): 用于判断异常的阈值
        """
        self.order = order
        self.threshold = threshold
        self.std_residuals = 0.0
        self.data = None
        self.fitted_values = None

    def fit(self, data):
        """
        拟合ARIMA模型
        
        Parameters:
        data (pd.Series): 输入的时间序列数据
        """
        self.data = data
        self.model = ARIMA(data, order=self.order)
        self.model = self.model.fit()
        self.fitted_values = self.model.fittedvalues

    def detect_outliers(self, data):
        """
        检测异常值
        
        Parameters:
        data (pd.Series): 输入的时间序列数据
        
        Returns:
        list: 检测到的异常值索引
        """
        if self.fitted_values is None:
            raise ValueError("模型尚未拟合，请先调用fit方法。")
        
        residuals = data - self.fitted_values
        # 因为对于处理GMV等数据较大的值,所以需要去掉residuals中的第一个值
        std_residuals = np.std(residuals[1:])
        outliers = np.where(np.abs(residuals) > self.threshold * std_residuals)[0]
        return outliers.tolist()

    # 根据AIC和BIC可以用于比较不同 ARIMA 模型（即不同的 (p, d, q) 参数组合）
    def get_best_pdq(self):
        # 定义 p, d, q 的范围
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)

        best_aic = float("inf")
        best_bic = float("inf")
        best_total_aic_bic = float("inf")
        best_order = None
        best_order_bic = None
        best_order_aic_bic = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        # 拟合 ARIMA 模型
                        model = ARIMA(self.data, order=(p, d, q))
                        model_fit = model.fit()

                        # 记录 AIC 和 BIC 值
                        aic = model_fit.aic
                        bic = model_fit.bic

                        total_aic_bic = model_fit.aic + model_fit.bic

                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)

                        if bic < best_bic:
                            best_bic = bic
                            best_order_bic = (p, d, q)

                        if total_aic_bic < best_total_aic_bic:
                            best_total_aic_bic = total_aic_bic
                            best_order_aic_bic = (p, d, q)

                        print(f"Order: ({p},{d},{q}) - AIC: {aic}, BIC: {bic}, AIC+BIC: {total_aic_bic}")
                    except:
                        continue

        print(f"Best AIC Order: {best_order}, Best AIC: {best_aic}")
        print(f"Best BIC Order: {best_order_bic}, Best BIC: {best_bic}")
        print(f"Best AIC+BIC Order: {best_order_aic_bic}, Best AIC+BIC: {best_order_aic_bic}")
        # 我们默选Best AIC+BIC
        return best_order_aic_bic

    # 检查输入的时间序列数据是否是稳定数据
    def check_stationarity(self):
        result = adfuller(self.data, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critical Values:')
            print(f'   {key}, {value}')
        if result[1] <= 0.05:
            print("Strong evidence against the null hypothesis, reject the null hypothesis.")
            print("Data has no unit root and is stationary.")
        else:
            print("Weak evidence against the null hypothesis, time series is non-stationary.")

    def is_outlier(self, test_data, label=""):
        if self.fitted_values is None:
            raise ValueError("模型尚未拟合，请先调用fit方法。")
        residuals = self.data - self.fitted_values
        # 因为对于处理GMV等数据较大的值,所以需要去掉residuals中的第一个值
        std_residuals = np.std(residuals[1:])
        new_data = self.model.forecast(steps=1)
        if label == "在营门店":
            if std_residuals < 50:
                std_residuals = 50
        self.std_residuals = std_residuals
        if abs(test_data - new_data[0]) <= self.threshold * std_residuals:
            return False
        return True

    def get_up_down_level(self, label=""):
        """
        找出数据中的所有上下区间。
        """
        residuals = self.data - self.fitted_values
        # 因为对于处理GMV等数据较大的值,所以需要去掉residuals中的第一个值
        std_residuals = np.std(residuals[1:])
        if label == "在营门店":
            if std_residuals < 50:
                std_residuals = 50
        new_data = self.model.forecast(steps=1)
        lower_bound = new_data[0] - self.threshold * std_residuals
        if lower_bound < 0.0:
            lower_bound = 0.0
        upper_bound = new_data[0] + self.threshold * std_residuals
        return round(lower_bound, 2), round(upper_bound, 2)

    def plot_results(self, data):
        """
        绘制结果
        
        Parameters:
        data (pd.Series): 输入的时间序列数据
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='原始数据', color='blue')
        plt.plot(self.fitted_values, label='拟合值', color='orange')
        plt.scatter(data.index[self.detect_outliers(data)], data[self.detect_outliers(data)], color='red', label='异常值')
        plt.legend()
        plt.title('ARIMA异常检测')
        # plt.show()

    def outlier_level(self, value, mean_temp, std_temp, label=""):
        outlier_level_score = 0.1
        outlier_level_describe = "需要关注"
        """
        异常值也要分层, 如果大于4Sigma，则为比较异常，3.5Sigma到4Sigma之间,则为中等异常, 3Sigma到3.5Sigma则为轻度异常
        """
        if label == "在营门店":
            new_data = self.model.forecast(steps=1)
            four_lower_bound = new_data - 6.0 * self.std_residuals
            four_upper_bound = new_data + 6.0 * self.std_residuals
            middle_lower_bound = mean_temp - 5.0 * self.std_residuals
            middle_upper_bound = mean_temp + 5.0 * self.std_residuals
            three_lower_bound = mean_temp - 4.0 * self.std_residuals
            three_upper_bound = mean_temp + 4.0 * std_temp
        else:
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


class OutlierDetectorWeightedAverage:
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else None
        self.data = None
        self.weighted_averages = None

    def fit(self, data, weights=None):
        self.data = np.array(data)
        self.weights = np.array(weights) if weights is not None else None
        if self.weights is None:
            self.weights = np.ones_like(self.data)
        self.weighted_averages = np.average(self.data, weights=self.weights)

    def detect_anomalies(self, threshold=3):
        deviations = np.abs(self.data - self.weighted_averages)
        standard_deviation = np.sqrt(np.average((deviations - np.average(deviations))**2, weights=self.weights))
        anomalies = np.where(deviations > threshold * standard_deviation)[0]
        return anomalies.tolist()

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


class OutlierDetectorEWMA:
    def __init__(self, span, alpha=0.2):
        self.span = span
        self.alpha = alpha
        self.ewma = None
        self.anomalies = None
        self.deviation_var = 0.0

    def fit(self, data):
        self.data = np.array(data)
        self.ewma = np.zeros_like(self.data)
        self.ewma[0] = self.data[0]
        for i in range(1, len(self.data)):
            self.ewma[i] = self.alpha * self.data[i] + (1 - self.alpha) * self.ewma[i-1]
        deviations = np.abs(self.data - self.ewma)
        self.deviation_var = deviations[1:].std()

    def detect_anomalies(self, threshold=3, label=""):
        deviations = np.abs(self.data - self.ewma)
        deviation_var = deviations[1:].std()
        self.anomalies = np.where(deviations > threshold * deviation_var)[0]
        # self.anomalies = np.where(deviations > threshold * np.mean(deviations))[0]
        return self.anomalies.tolist()

    def is_outlier(self, data, threshold=3, label=""):
        data_new = self.alpha * data + (1 - self.alpha) * self.ewma[-1]
        deviation = np.abs(data - data_new)
        deviation_var = self.deviation_var
        if label == "在营门店":
            if deviation_var <= 50:
                # 稍微扩大范围
                deviation_var = 50
        if deviation <= deviation_var * threshold:
            return False
        return True

    def get_up_down_level(self, value, threshold, label=""):
        """
        找出数据中的所有上下区间。
        """
        deviation_var = self.deviation_var
        if label == "在营门店":
            if deviation_var <= 50:
                # 稍微扩大范围
                deviation_var = 50
        data_new = self.alpha * value + (1 - self.alpha) * self.ewma[-1]
        lower_bound = data_new - threshold * deviation_var
        if lower_bound < 0.0:
            lower_bound = 0.0
        upper_bound = data_new + threshold * deviation_var
        return round(lower_bound, 2), round(upper_bound, 2)

    def outlier_level(self, value, mean_temp, std_temp, label=""):
        """
        异常值也要分层, 如果大于4Sigma，则为比较异常，3.5Sigma到4Sigma之间,则为中等异常, 3Sigma到3.5Sigma则为轻度异常
        """
        outlier_level_score = 0.1
        outlier_level_describe = "需要关注"
        if label == "在营门店":
            data_new = self.alpha * value + (1 - self.alpha) * self.ewma[-1]
            deviation_var = self.deviation_var
            if deviation_var <= 50:
                # 稍微扩大范围
                deviation_var = 50
            four_lower_bound = data_new - 6.0 * deviation_var
            four_upper_bound = data_new + 6.0 * deviation_var
            middle_lower_bound = data_new - 5.0 * deviation_var
            middle_upper_bound = data_new + 5.0 * deviation_var
            three_lower_bound = data_new - 4.0 * deviation_var
            three_upper_bound = data_new + 4.0 * deviation_var
        else:
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


class OutlierDetectorGrubbsTest:
    def __init__(self, alpha=0.05):
        self.alpha = alpha# 设置显著性水平

    def calculate_critical_value(self, n):#  计算Grubbs' Test的临界值，基于样本大小n和显著性水平alpha
        t_value = t.ppf(1 - self.alpha / (2 * n), n - 2)
        critical_value = ((n - 1) * np.sqrt(t_value**2)) / (np.sqrt(n) * np.sqrt(n - 2 + t_value**2))
        return critical_value

    def test(self, data):
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        max_deviation = np.max(np.abs(data - mean))
        grubbs_statistic = max_deviation / std_dev
        critical_value = self.calculate_critical_value(n)

        if grubbs_statistic > critical_value:
            outlier_index = np.argmax(np.abs(data - mean))
            return True, data[outlier_index]
        else:
            return False, None

    def get_up_down_level(self, data):
        """
        找出数据中的所有上下区间。
        """
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        critical_value = self.calculate_critical_value(n)
        lower_bound = mean - critical_value * std_dev
        if lower_bound < 0.0:
            lower_bound = 0.0
        upper_bound = mean + critical_value * std_dev
        return round(lower_bound, 2), round(upper_bound, 2)

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


class OutlierDetectorIForest:
    def __init__(self, n_estimators=100, contamination=0.1, random_state=None):
        """
        初始化Isolation Forest异常检测器。
        
        参数:
        - n_estimators: int, 可选 (default=100)
          用于构建森林的树的数量。
        - contamination: float, 可选 (default=0.1)
          数据集中异常值的比例。
        - random_state: int, 可选 (default=None)
          随机数生成器的种子。
        """
        self.model = IsolationForest(n_estimators=n_estimators, 
                                     contamination=contamination, 
                                     random_state=random_state)

    def fit(self, X):
        """
        拟合模型。
        
        参数:
        - X: array-like, shape (n_samples, n_features)
          训练数据。
        """
        self.model.fit(X)

    def predict(self, X):
        """
        对新数据进行异常检测。
        
        参数:
        - X: array-like, shape (n_samples, n_features)
          输入数据。
        
        返回:
        - array, shape (n_samples,)
          每个样本的预测标签（1表示正常，-1表示异常）。
        """
        return self.model.predict(X)

    def anomaly_score(self, X):
        """
        计算异常分数。
        计算每个样本的异常分数，分数越低表示越可能是异常值。
        参数:
        - X: array-like, shape (n_samples, n_features)
          输入数据。
        
        返回:
        - array, shape (n_samples,)
          每个样本的异常分数。
        """
        return self.model.decision_function(X)

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