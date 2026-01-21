import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class SalesForecast:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """准备特征数据"""
        # 时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        
        # 季节性特征
        df['is_holiday'] = self.is_holiday(df['date'])  # 假期标记
        
        # 商品特征
        df = pd.get_dummies(df, columns=['product_category'])
        
        return df
        
    def train(self, historical_data):
        """训练模型"""
        # 准备特征
        X = self.prepare_features(historical_data)
        y = historical_data['sales']
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mse, r2
    
    def predict(self, future_data):
        """预测未来销量"""
        X = self.prepare_features(future_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @staticmethod
    def is_holiday(dates):
        """判断是否为节假日"""
        # 这里需要导入节假日数据或使用相关API
        # 示例实现
        return np.zeros(len(dates))


if __name__ == "__main__":
    # 准备历史数据
    historical_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2024-03-01'),
        'product_category': ['A', 'B', 'C'] * 100,  # 示例产品类别
        'sales': np.random.randint(50, 200, 300),   # 示例销量数据
        # 其他特征...
    })

    # 初始化预测器
    forecaster = SalesForecast()

    # 训练模型
    mse, r2 = forecaster.train(historical_data)
    print(f"模型评估 - MSE: {mse:.2f}, R2: {r2:.2f}")

    # 预测未来销量
    future_data = pd.DataFrame({
        'date': pd.date_range(start='2024-03-02', end='2024-04-01'),
        'product_category': ['A', 'B', 'C'] * 10,
        # 其他特征...
    })

    predictions = forecaster.predict(future_data)