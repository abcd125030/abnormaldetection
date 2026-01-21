import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import TIMESTAMP

from api.script.anosvgd import AnoSVGD
from api.script.statistic import OutlierDetectorThreeSigma, OutlierDetectorBoxPlot, OutlierDetectorGrubbsTest, \
    OutlierDetectorWeightedAverage, OutlierDetectorEWMA, \
    OutlierDetectorARIMA
from config.conf import get_normal_interval_from_mysql_adapter, database_config, outlier_level, get_week, \
    is_monthly_data, is_within_35_days_before_date, check_special_date
from config.database_class import DbAdapter
from config.database_pymysql import DbSqlalchemy


# 目前kl这个测试后发现效果不好，所以就先不添加了
# vbem目前没有实现
def my_algorithm(data):
    # 算法实现 Generate synthetic data
    # np.random.seed(42)
    algorithm_begin_time = time.time()
    outlier_results = list()
    try:
        # 用户可以选择检测算法
        detect_type_input = data['检测算法']
        # 目前先只处理3sigma, Boxplot, Grubbs
        if detect_type_input != "3sigma" and detect_type_input != "boxplot" and detect_type_input != "grubbs" and detect_type_input != "weighted_avg":
            detect_type_input = "3sigma"
        logger.bind(decorChaGee=True).info("detect_type_input: " + detect_type_input)
        # 监控项(同比, 环比, 基线：基于统计周期内的统计值, 基线：基于统计周期内的预测值)
        # monitor_item = data['监控项']
        # 是否需要检测传入数据是否有异常
        bSaveInputToDatabase = True
        if "是否将输入数据保存到数据库中" in data:
            bSaveInputToDatabase = data['是否将输入数据保存到数据库中']
        if bSaveInputToDatabase:
            host, port, username, password, database = database_config()
            dbSqlalchemy = DbSqlalchemy()
            dbSqlalchemy.dbSetup(host, port, username, password, database)
            # 根据参数传入来获取每个参数的区间上下限
            input_data = json.dumps(data, ensure_ascii=False)
            input_msyql_data = {
                'process_time': [datetime.now()],
                'process_datas': [input_data]
            }
            try:
                # 将DataFrame写入数据库
                df = pd.DataFrame(input_msyql_data)
                df.to_sql('om_v3_abnormal_detection_input', con=dbSqlalchemy.engine, index=False, if_exists='append',
                          dtype={'process_time': TIMESTAMP})
                logger.bind(decorHaier=True).info('写入sql成功')
            except Exception as e:
                logger.bind(decorHaier=True).info('写入sql错误，%s' % (e))

        bCheckFirst = True
        if "是否需要检测传入数据是否有异常" in data:
            bCheckFirst = data['是否需要检测传入数据是否有异常']
        # 是否使用已有的上下限值来判断异常
        bReadDatabase = False
        cDbAdapter = DbAdapter()
        if "是否使用已有的上下限值来判断异常" in data:
            bReadDatabase = data['是否使用已有的上下限值来判断异常']
            if bReadDatabase:
                host, port, username, password, database = database_config()
                cDbAdapter.dbSetup(host, port, username, password, database)

        for label_info in data['指标']:
            # 检测的指标值
            label_name = ""
            # 需要检测的值
            data_input_value = 0.0
            data_input_his_dict = dict()
            data_input_his_values = list()
            for key in label_info.keys():
                if key == "指标名称":
                    label_name = label_info[key]
                elif key == "历史指标数据":
                    data_input_his_dict = label_info[key]
                else:
                    data_input_value = float(label_info[key])

            # 目前让"运营门店"强制走"arima"检测算法
            if label_name == "在营门店":
                detect_type = "arima"
                # detect_type = "ewma"
                logger.bind(decorChaGee=True).info("detect_type change to : " + detect_type)
            else:
                detect_type = detect_type_input
            # 检测的指标值
            if bReadDatabase:
                source = "核心指标"
                if "指标板块" in data:
                    source = data['指标板块']
                input_period = data['日期']
                input_hour = data['小时']
                input_dem = data['维度']
                hour = -1
                week = -1
                month = -1
                if input_dem == '当日':
                    hour = int(input_hour.split("-")[0])
                if input_dem == '当周':
                    week = get_week(input_period)
                    hour = int(input_hour.split("-")[0])
                if input_dem == '当月':
                    month = int(input_period.split("-")[2])
                    hour = int(input_hour.split("-")[0])

                lower_range, upper_range, mean, variance = get_normal_interval_from_mysql_adapter(source, input_dem, label_name, detect_type, month, week, hour, cDbAdapter)
                if data_input_value < lower_range or data_input_value > upper_range:
                    outlier_level_score, outlier_level_describe = outlier_level(data_input_value, mean, variance)
                    outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_range,
                                      "上限区间": upper_range, "均值": mean,
                                      "方差": variance, "异常率": outlier_level_score, "异常程度": outlier_level_describe,
                                      "status": 0}
                    logger.bind(decorChaGee=True).info(outlier_result)
                else:
                    outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_range,
                                      "上限区间": upper_range, "均值": mean,
                                      "方差": variance, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                    logger.bind(decorChaGee=True).info(outlier_result)
                outlier_results.append(outlier_result)
            else:
                bmonthly = is_monthly_data(data_input_his_dict)
                bdialy = is_within_35_days_before_date(data_input_his_dict, data['日期'])
                # 历史数据
                # 因为当日，昨日的数据的日期不规则，可能会影响arima算法，所以需要按照日期处理
                data_input_his_values = [value for key, value in sorted(data_input_his_dict.items(), key=lambda item: item[0])]
                # for key in data_input_his_dict.keys():
                #     data_input_his_values.append(float(data_input_his_dict[key]))

                logger.bind(decorChaGee=True).info("----检测的指标值:----" + label_name)
                if bCheckFirst:
                    # 先处理一下传入的历史值的异常，将异常值删除
                    logger.bind(decorChaGee=True).info("----处理传入的历史值的异常，将异常值删除开始----")
                    # 创建 OutlierDetectorBoxPlot 类的实例
                    detector = OutlierDetectorBoxPlot()
                    # 检测异常值
                    detector.detect_anomalies(data_input_his_values)
                    # 获取异常值
                    anomalies = detector.get_anomalies()
                    # 将小列表转换为集合
                    anomalies_set = set(anomalies)
                    # 使用集合的差集操作
                    removed_result = [x for x in data_input_his_values if x not in anomalies_set]
                    # 异常值删除结束
                    logger.bind(decorChaGee=True).info("----处理传入的历史值的异常，将异常值删除结束----")
                else:
                    removed_result = data_input_his_values

                # # 如果要计算环比， 将data_input_value和removed_result转为环比
                # if monitor_item == "环比":
                #     pass

                # 计算传入数据的均值和方差
                mean = np.mean(removed_result)
                dev = np.std(removed_result)
                # 开始检测
                if detect_type == '3sigma':
                    # 3sigma
                    # 基线：基于统计周期内的统计值
                    detector_threshold = 3.0
                    # 要考虑节假日可能超过3sigma的上限
                    over_3sigma_threshold_temp = -1.0
                    # 新增门店可能超过3sigma的下限
                    lower_3sigma_threshold_temp = -1.0
                    # 如果是这段时间是节假日，或是上个月是节假日，则上限值设为4sigma
                    if label_name == "gmv" or label_name == "出杯量" or label_name == "成交订单":
                        # 目前主要是处理月度数据
                        if bmonthly:
                            over_3sigma_threshold_temp = 6.0
                            lower_3sigma_threshold_temp = 6.0
                        elif bdialy:
                            result = check_special_date()
                            if result:
                                logger.bind(decorChaGee=True).info(f"今天是：{result}")
                                over_3sigma_threshold_temp = 7.0
                            else:
                                logger.bind(decorChaGee=True).info("今天不是特殊节日")
                    if label_name == "新增会员":
                        over_3sigma_threshold_temp = 6.0
                        lower_3sigma_threshold_temp = 6.0
                    if label_name == "新增门店":
                        # 允许当日的新增门店出现0
                        if not bdialy:
                            removed_result = [item for item in removed_result if item != 0.0]
                        mean = np.mean(removed_result)
                        dev = np.std(removed_result)
                        # "历史指标数据": {
                        #     "2024-07-29 16H": 0,
                        #     "2024-08-05 16H": 0,
                        #     "2024-08-12 16H": 0,
                        #     "2024-08-19 16H": 0,
                        #     "2024-08-26 16H": 0,
                        #     "2024-09-02 16H": 0,
                        #     "2024-09-09 16H": 0,
                        #     "2024-09-23 16H": 0,
                        #     "2024-09-30 16H": 0,
                        #     "2024-10-07 16H": 0,
                        #     "2024-10-14 16H": 0,
                        #     "2024-10-21 16H": 0,
                        #     "2024-10-28 16H": 0,
                        #     "2024-11-04 16H": 0,
                        #     "2024-11-11 16H": 0,
                        #     "2024-11-18 16H": 0,
                        #     "2024-11-25 16H": 0,
                        #     "2024-12-02 16H": 0,
                        #     "2024-12-09 16H": 0,
                        #     "2024-12-16 16H": 0,
                        #     "2024-12-23 16H": 4,
                        #     "2024-12-30 16H": 3,
                        #     "2025-01-06 16H": 3
                        # }
                        # 临时处理一下
                        if mean <= 10.0:
                            if dev < 1.0:
                                over_3sigma_threshold_temp = 20.0
                                lower_3sigma_threshold_temp = 20.0
                            else:
                                over_3sigma_threshold_temp = 10.0
                                lower_3sigma_threshold_temp = 10.0

                    removed_detector = OutlierDetectorThreeSigma(removed_result, detector_threshold, over_3sigma_threshold_temp, lower_3sigma_threshold_temp)
                    lower_bound, upper_bound = removed_detector.get_up_down_level()
                    if removed_detector.is_outlier(data_input_value):
                        outlier_level_score, outlier_level_describe = removed_detector.outlier_level(data_input_value, mean, dev)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound, "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound, "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                elif detect_type == 'boxplot':
                    # 箱线图(Box Plot)
                    # 基线：基于统计周期内的统计值
                    detector = OutlierDetectorBoxPlot()
                    detector_threshold = 1.5
                    detector.detect_anomalies(removed_result, detector_threshold)
                    lower_bound, upper_bound = detector.get_up_down_level()
                    if detector.detect_anomaly_data(data_input_value):
                        outlier_level_score, outlier_level_describe = detector.outlier_level(data_input_value, mean, dev)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score,  "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                elif detect_type == "grubbs":
                    # 格鲁布斯检验(Grubbs' Test适用于正态分布的数据，并且一次只能检测一个异常值)
                    # 基线：基于统计周期内的统计值
                    alpha = 0.5
                    grubbs_test = OutlierDetectorGrubbsTest(alpha)
                    # 将本次需要检测的值也加入removed_result中
                    removed_result.append(data_input_value)
                    lower_bound, upper_bound = grubbs_test.get_up_down_level(removed_result)
                    is_outlier, outlier_value = grubbs_test.test(removed_result)
                    if is_outlier:
                        logger.bind(decorChaGee=True).info(f"检测到异常值: {outlier_value}")
                        if data_input_value == outlier_value:
                            outlier_level_score, outlier_level_describe = grubbs_test.outlier_level(data_input_value, mean, dev)
                            outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                              "上限区间": upper_bound, "均值": mean,
                                              "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                            logger.bind(decorChaGee=True).info(outlier_result)
                        else:
                            outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                              "上限区间": upper_bound, "均值": mean,
                                              "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                            logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                elif detect_type == 'weighted_avg':
                    # 加权平均（Weighted Average）
                    # 基线：基于统计周期内的统计值
                    # 将本次需要检测的值也加入removed_result中
                    # removed_result.append(data_input_value)
                    # weights = np.ones(len(removed_result), dtype=int)
                    # # 创建异常检测器实例
                    # lower_bound = -1
                    # upper_bound = -1
                    # detector = OutlierDetectorWeightedAverage(weights=weights)
                    # # 训练模型
                    # detector.fit(removed_result, weights)
                    # # 检测异常值
                    # detector_threshold = 3.0
                    # anomalies = detector.detect_anomalies(detector_threshold)
                    # anomaly_datas = [removed_result[x] for x in anomalies]
                    # if data_input_value in anomaly_datas:
                    #     outlier_level_score, outlier_level_describe = detector.outlier_level(data_input_value, mean, dev)
                    #     outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                    #                       "上限区间": upper_bound, "均值": mean,
                    #                       "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                    #     logger.bind(decorChaGee=True).info(outlier_result)
                    # else:
                    #     outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                    #                       "上限区间": upper_bound, "均值": mean,
                    #                       "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                    #     logger.bind(decorChaGee=True).info(outlier_result)
                    # 临时处理
                    detector_threshold = 2.5
                    removed_detector = OutlierDetectorThreeSigma(removed_result, detector_threshold)
                    lower_bound, upper_bound = removed_detector.get_up_down_level()
                    if removed_detector.is_outlier(data_input_value):
                        outlier_level_score, outlier_level_describe = removed_detector.outlier_level(data_input_value,
                                                                                                     mean, dev)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score,
                                          "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                elif detect_type == 'anosvgd':
                    # AnoSVGD（Anomaly Detection using Stein Variational Gradient Descent）
                    # 是一种结合了Stein Variational Gradient Descent (SVGD) 和异常检测的方法
                    # 基线：基于统计周期内的预测值
                    # 将本次需要检测的值也加入removed_result中
                    # 结果和随着初始粒子位置有关, 所以再临界处的值可能被认为是异常，也可能不被认为是异常
                    removed_result.append(data_input_value)
                    lower_bound = -1
                    upper_bound = -1
                    # 训练模型
                    x_train = np.array(removed_result).reshape(-1, 1)
                    # 创建AnoSVGD实例
                    anosvgd = AnoSVGD(n_particles=100, lr=0.01, n_iter=1000, kernel_bandwidth=1.0)
                    # 拟合数据
                    anosvgd.fit(x_train)
                    # 检测异常值
                    anomaly_datas = anosvgd.detect(x_train, threshold_percentile=95)
                    if data_input_value in anomaly_datas:
                        outlier_level_score, outlier_level_describe = anosvgd.outlier_level(data_input_value, mean, dev)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                elif detect_type == 'ewma':
                    # # EWMA（Exponentially Weighted Moving Average，指数加权移动平均）
                    # # 基线：基于统计周期内的预测值
                    # # 创建EWMA异常检测器实例，设定时间跨度为10，alpha为0.1
                    # # 默认用户传入的值是按照时间序列的
                    # detector = OutlierDetectorEWMA(span=len(removed_result), alpha=0.1)
                    # # 将本次需要检测的值也加入removed_result中
                    # removed_result.append(data_input_value)
                    # # 训练模型
                    # x_train = np.array(removed_result)
                    # detector.fit(x_train)
                    # # 检测异常值
                    # anomalies = detector.detect_anomalies(threshold=3)
                    # lower_bound, upper_bound = -1, -1
                    # anomaly_datas = [removed_result[x] for x in anomalies]
                    # if data_input_value in anomaly_datas:
                    #     outlier_level_score, outlier_level_describe = detector.outlier_level(data_input_value, mean, dev)
                    #     outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                    #                       "上限区间": upper_bound, "均值": mean,
                    #                       "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                    #     logger.bind(decorChaGee=True).info(outlier_result)
                    # else:
                    #     outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                    #                       "上限区间": upper_bound, "均值": mean,
                    #                       "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                    #     logger.bind(decorChaGee=True).info(outlier_result)
                    # EWMA（Exponentially Weighted Moving Average，指数加权移动平均）
                    # 基线：基于统计周期内的预测值
                    # 创建EWMA异常检测器实例，设定时间跨度为10，alpha为0.1
                    # 默认用户传入的值是按照时间序列的
                    detector = OutlierDetectorEWMA(span=len(removed_result), alpha=0.5)
                    # 训练模型
                    x_train = np.array(removed_result)
                    detector.fit(x_train)
                    # 检测异常值
                    lower_bound, upper_bound = detector.get_up_down_level(data_input_value, threshold=3, label=label_name)
                    if detector.is_outlier(data_input_value, threshold=3, label=label_name):
                        outlier_level_score, outlier_level_describe = detector.outlier_level(data_input_value, mean,
                                                                                             dev, label=label_name)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score,
                                          "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                elif detect_type == 'arima':
                    # # ARIMA（AutoRegressive Integrated Moving Average，自回归积分滑动平均模型）
                    # # 基线：基于统计周期内的预测值
                    # detector = OutlierDetectorARIMA(order=(1, 1, 1), threshold=2)
                    # # 将本次需要检测的值也加入removed_result中
                    # # removed_result.append(data_input_value)
                    # lower_bound = -1
                    # upper_bound = -1
                    # # 训练模型
                    # x_train = np.array(removed_result)
                    # detector.fit(x_train)
                    # # 检测异常值
                    # anomalies = detector.detect_outliers(removed_result)
                    # anomaly_datas = [removed_result[x] for x in anomalies]
                    # if data_input_value in anomaly_datas:
                    #     outlier_level_score, outlier_level_describe = detector.outlier_level(data_input_value, mean, dev)
                    #     outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                    #                       "上限区间": upper_bound, "均值": mean,
                    #                       "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                    #     logger.bind(decorChaGee=True).info(outlier_result)
                    # else:
                    #     outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                    #                       "上限区间": upper_bound, "均值": mean,
                    #                       "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                    #     logger.bind(decorChaGee=True).info(outlier_result)
                    detector = OutlierDetectorARIMA(order=(1, 1, 1), threshold=3)
                    # 训练模型
                    x_train = np.array(removed_result)
                    detector.fit(x_train)
                    # 检测异常值
                    lower_bound, upper_bound = detector.get_up_down_level(label=label_name)
                    if detector.is_outlier(data_input_value, label=label_name):
                        outlier_level_score, outlier_level_describe = detector.outlier_level(data_input_value, mean,
                                                                                             dev, label=label_name)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound,
                                          "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                else:
                    # 如果不传入异常检测方式的话,默认走的是3sigma
                    logger.bind(decorChaGee=True).info(f'The default anomaly detection algorithm is : 3sigma')
                    detector_threshold = 3.0
                    removed_detector = OutlierDetectorThreeSigma(removed_result, detector_threshold)
                    lower_bound, upper_bound = removed_detector.get_up_down_level()
                    if removed_detector.is_outlier(data_input_value):
                        outlier_level_score, outlier_level_describe = removed_detector.outlier_level(data_input_value, mean, dev)
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound, "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": outlier_level_score, "异常程度": outlier_level_describe, "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)
                    else:
                        outlier_result = {"指标名称": label_name, "指标数据": data_input_value, "下限区间": lower_bound, "上限区间": upper_bound, "均值": mean,
                                          "方差": dev, "异常率": 0.0, "异常程度": "没有异常", "status": 0}
                        logger.bind(decorChaGee=True).info(outlier_result)

                outlier_results.append(outlier_result)
    except Exception as e:
        logger.bind(decorChaGee=True).info(f'parse input and select detect type with error:{e}')
        outlier_results.append({"异常率": 1.0, "status": -1, "出错原因": str(e)})
    finally:
        algorithm_end_time = time.time()
        test_time_dict = {"异常检测所需时间": round(algorithm_end_time - algorithm_begin_time, 3)}
        return outlier_results, test_time_dict
