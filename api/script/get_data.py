# 获取om_v3_result_items和om_v3_his_result_items中datas_json中的内容
import datetime
import json
import os

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from api.script import path_algorithm
from config import settings
from config.database_class import DbAdapter


def database_config():
    config = settings
    db_config = config['database-prod']
    host = db_config['host']
    port = db_config['port']
    user = db_config['user']
    password = db_config['password']
    database = db_config['database']
    logger.bind(decorChaGee=True).info(
        "host-{}, port-{}, user-{}, password-{}, database-{}:".format(host, port, user, password, database))
    return host, port, user, password, database


def get_last_month(process_time):
    # 获取当前月份的第一天
    first_day_of_this_month = process_time.replace(day=1)
    # 获取上个月的最后一天
    last_day_of_last_month = first_day_of_this_month - datetime.timedelta(days=1)
    # 获取上个月的月份
    last_month_year, last_month = last_day_of_last_month.year, last_day_of_last_month.month
    print("前一月是 {} 年{} 月".format(last_month_year, last_month))
    return last_month_year, last_month


def get_cur_month(process_time):
    cur_year, cur_month = process_time.year, process_time.month
    return cur_year, cur_month

def get_last_day(process_time):
    yesterday = process_time - datetime.timedelta(days=1)
    return yesterday

def get_last_week(process_time):
    # 获取当前日期是本周的第几天（周一为1，周日为7）
    weekday = process_time.isoweekday()
    # 获取本周一的日期
    this_monday = process_time - datetime.timedelta(days=weekday - 1)
    # 获取上周一的日期
    last_monday = this_monday - datetime.timedelta(days=7)
    # 获取前一周的年份和周数
    year, week_number, _ = last_monday.isocalendar()
    print("前一周是 {} 年的第 {} 周".format(year, week_number))
    return year, week_number


def getcontent(data_json_str, period, source):
    content = dict()
    if data_json_str:
        data_json = json.loads(data_json_str)
        for data_temp in data_json:
            if data_temp['周期'] == period and data_temp['来源'] == source:
                for key in data_temp.keys():
                    if key != "周期":
                        content[key] = data_temp[key]
    return content


def plot_result(df_dataX, df_dataY, plt_title, plt_xlabel, plt_ylabel):
    # 绘制训练数据、测试数据和预测结果
    plt.figure(figsize=(14, 7))
    plt.title(plt_title)
    plt.plot(df_dataX, df_dataY, 'ro', markersize=10)
    plt.plot(df_dataX, df_dataY, label='Line', linestyle='--', color='orange')
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def Supplement_missing_information(day_info_data, begin_day, end_day):
    # 定义起始日期和结束日期
    day_info_dat_new = dict()
    date_range = pd.date_range(start=begin_day, end=end_day)
    # 遍历日期范围
    for date in date_range:
        print(date.strftime('%Y-%m-%d'))
        if not date.strftime('%Y-%m-%d') in day_info_data:
            # 根据蔡佳的建议,先将缺失数据的日期用7天前的数据弥补
            day_info_dat_new[date.strftime('%Y-%m-%d')] = {"gmv": "0.0", "实收": "0.0", "成交订单": "0.0", "新增会员": "0.0", "累计会员": "0.0", "品牌指数": "0.0", "出杯量": "0.0", "新增门店": "0.0", "在营门店": "0.0"}
        else:
            day_info_dat_new[date.strftime('%Y-%m-%d')] = day_info_data[date.strftime('%Y-%m-%d')]
    return day_info_dat_new


def get_his_values(label_type):
    all_his_values = []
    path_algorithm_parent = os.path.dirname(path_algorithm)
    file_path = path_algorithm_parent + '/data/input/national_daily_data.json'

    # 打开文件并读出JSON数据
    with open(file_path, 'r', encoding='utf-8') as json_file:
        day_info_dat_new = json.load(json_file)

    for day_key in day_info_dat_new.keys():
        all_his_values.append(float(day_info_dat_new[day_key][label_type]))
    return all_his_values


if __name__ == "__main__":
    cDbAdapter = DbAdapter()
    host, port, username, password, database = database_config()
    cDbAdapter.dbSetup(host, port, username, password, database)
    sql_word = """SELECT
                id,
                process_time,
                freq_code,
                datas_json
            FROM
                (
                    SELECT
                        x.id,
                        process_time,
                        x.freq_code,
                        y.datas_json
                    FROM
                        om_v3_result X
                        LEFT JOIN om_v3_result_items Y ON x.id = y.result_id
                    WHERE
                        config_id = 9
                        AND y.datas_json LIKE '%来源":"趋势图%'
                        AND HOUR (process_time) >= 0
                        AND HOUR (process_time) < 2
                    UNION
                    SELECT
                        x.id,
                        process_time,
                        x.freq_code,
                        y.datas_json
                    FROM
                        om_v3_his_result X
                        LEFT JOIN om_v3_his_result_items Y ON x.id = y.result_id
                    WHERE
                        config_id = 9
                        AND y.datas_json LIKE '%来源":"趋势图%'
                        AND MONTH (process_time) = 7
                        AND DAY (process_time) = 30
                        AND HOUR (process_time) >= 23
                    UNION
                    SELECT
                        x.id,
                        process_time,
                        x.freq_code,
                        y.datas_json
                    FROM
                        om_v3_his_result X
                        LEFT JOIN om_v3_his_result_items Y ON x.id = y.result_id
                    WHERE
                        config_id = 9
                        AND y.datas_json LIKE '%来源":"趋势图%'
                        AND HOUR (process_time) >= 0
                        AND HOUR (process_time) < 2
                ) AS temp
            ORDER BY
                process_time"""

    results = cDbAdapter.dbQuery(sql_word)
    if len(results) > 0:
        day_info_data = dict()
        week_info_data = dict()
        month_info_data = dict()

        for result in results:
            process_time = result['process_time'].strftime('%Y-%m-%d')
            print(process_time)
            last_month_year, last_month = get_last_month(result['process_time'])
            last_week_year, last_week = get_last_week(result['process_time'])
            last_day = get_last_day(result['process_time']).strftime('%Y-%m-%d')
            print("get_last_day: " + last_day)

            if not last_day in day_info_data:
                day_info_data[last_day] = getcontent(result['datas_json'], "昨日", "趋势图")

            if not (str(last_week_year) + "-" + str(last_week)) in week_info_data:
                week_info_data[str(last_week_year) + "-" + str(last_week)] = getcontent(result['datas_json'], "上周", "趋势图")

            if not (str(last_month_year) + "-" + str(last_month)) in month_info_data:
                month_info_data[str(last_month_year) + "-" + str(last_month)] = getcontent(result['datas_json'], "上月", "趋势图")

        # 缺失2024-09-14到2024-09-20的数据
        # 需要修补，如何修补目前需要讨论，最好让李延民补充。不能的话，我们用前一周的数据顶替(或者按照星期取平均值)
        begin_day = "2024-07-25"
        end_day = "2024-11-19"
        day_info_dat_new = Supplement_missing_information(day_info_data, begin_day, end_day)

        print('数据整理完毕')
        # # 开始画图
        # plt_title = "day_info_dat_new + GMV"
        # plt_xlabel = "time"
        # plt_ylabel = "gmv"
        # df_dataX = list()
        # df_dataY = list()
        # # df_dataX = ["2024-11-11", "2024-11-12", "2024-11-13"]
        # # df_dataY = [100, 200, 150]
        # for key in day_info_dat_new.keys():
        #     df_dataX.append(key)
        #     df_dataY.append(int(float(day_info_dat_new[key][plt_ylabel])))
        # plot_result(df_dataX, df_dataY, plt_title, plt_xlabel, plt_ylabel)

        # 将全国的GMV等数据保存成json文件
        # 指定保存的文件路径
        # file_path = '../data/input/national_daily_data.json'
        #
        # # 打开文件并写入 JSON 数据
        # with open(file_path, 'w', encoding='utf-8') as json_file:
        #     json.dump(day_info_dat_new, json_file, ensure_ascii=False, indent=4)
        #
        # print(f"数据已成功保存到 {file_path}")

        file_path = '../data/input/national_daily_data_gmv.json'

        day_info_dat_new_gmv = dict()
        for key in day_info_dat_new.keys():
            day_info_dat_new_gmv[key] = day_info_dat_new[key]['gmv']

        # 打开文件并写入 JSON 数据
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(day_info_dat_new_gmv, json_file, ensure_ascii=False, indent=4)

        print(f"数据已成功保存到 {file_path}")


