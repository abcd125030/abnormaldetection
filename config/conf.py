# -*- coding: utf-8 -*-
# @Time : 2024/11/12 16:36
# @File : conf.py
# @Author : lisongming
import datetime
import json
import os
from decimal import Decimal
import numpy as np
from loguru import logger
from config import settings


if not os.path.exists("./logs"):
    os.makedirs("./logs")

logger.bind(decorChaGee=True).add(
    './logs/decorChaGee_{time}.logs',
    filter=lambda record: "decorChaGee" in record["extra"],
    format="{time:YYYY-MM-DD at HH:mm:ss} | process_id:{process.id} process_name:{process.name} | {level: <8} | {name: ^15} | {function: ^15} | {line: >3} | {message}",
    colorize=True,
    rotation='1 week',
    retention='1 months',
    encoding='utf-8',
    enqueue=True,
    backtrace=True,
    diagnose=True
)


class DBConfig(object):
    def __init__(self):
        # 海尔预发
        host, port, username, password, database = database_config()
        self.DB_ACCOUNT = username  # 用户名
        self.DB_PWD = password  # 密码
        self.DB_HOST = host  # IP地址
        self.DB_PORT = port  # 端口
        self.DB_NAME = database  # 数据库名称
        self.DB_POOL_RECYCLE = True  # 多久之后对连接池中连接进行一次回收
        self.DB_MAX_OVERFLOW = 200  # 超过连接池大小之后，允许最大扩展连接数
        self.DB_POOL_SIZE = 128  # 连接池的大小
        self.DB_POOL_TIMEOUT = 10  # 连接池如果没有连接了，最长的等待时间

    def print_db(self):
        logger.bind(decorHaier=True).info("DB_ACCOUNT = " + self.DB_ACCOUNT)
        logger.bind(decorHaier=True).info("DB_PWD = " + self.DB_PWD)
        logger.bind(decorHaier=True).info("DB_HOST = " + self.DB_HOST)
        logger.bind(decorHaier=True).info("DB_PORT = " + str(self.DB_PORT))
        logger.bind(decorHaier=True).info("DB_NAME = " + self.DB_NAME)


# 修改
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


def getcontent(data_json_str, period, source, bPeriod):
    content = dict()
    if data_json_str:
        data_json = json.loads(data_json_str)
        for data_temp in data_json:
            if data_temp['周期'] == period and data_temp['来源'] == source:
                for key in data_temp.keys():
                    if bPeriod:
                        content[key] = data_temp[key]
                    else:
                        if key != "周期":
                            content[key] = data_temp[key]
    return content


# 获取当前日期是本周的第几天（周一为1，周日为7）
def get_week(process_time):
    date_format = "%Y-%m-%d"  # 年-月-日
    date_object = datetime.datetime.strptime(process_time, date_format)
    weekday = date_object.isoweekday()
    return weekday


def get_normal_interval_from_mysql_adapter(source, period, label, algorithm, monthday, weekday, hour, cDbAdapter):
    sql_word = "SELECT lower_range, upper_range, mean, variance FROM om_v3_abnormal_detection_test WHERE SOURCE = '{source}' AND PERIOD = '{period}' " \
               "AND ALGORITHM = '{algorithm}' AND label = '{label}' AND monthday = {monthday} AND weekday = {weekday} AND hour = {hour}".\
        format(source=source, period=period, label=label, algorithm=algorithm, monthday=monthday, weekday=weekday, hour=hour)

    results = cDbAdapter.dbQuery(sql_word)
    lower_range = -1.0
    upper_range = -1.0
    mean = -1.0
    variance = -1.0
    if len(results) == 1:
        lower_range = round(float(results[0]['lower_range']), 2)
        upper_range = round(float(results[0]['upper_range']), 2)
        mean = round(float(results[0]['mean']), 2)
        variance = round(float(results[0]['variance']), 2)
    return lower_range, upper_range, mean, variance


def outlier_level(value, mean_temp, std_temp):
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


# 判断是不是月度
def is_monthly_data(data):
    # 解析时间戳并排序
    timestamps = sorted([datetime.datetime.strptime(key, "%Y-%m-%d %HH") for key in data.keys()])

    # 检查是否有至少两个时间戳以进行比较
    if len(timestamps) < 2:
        return False

    # 检查所有时间戳是否都在每个月的同一天同一时间
    day_of_month = timestamps[0].day
    hour_of_day = timestamps[0].hour

    for ts in timestamps:
        if ts.day != day_of_month or ts.hour != hour_of_day:
            return False

    return True


# 判断是不是周度
def is_weekly_data(data):
    # 解析时间戳并排序
    timestamps = sorted([datetime.datetime.strptime(key, "%Y-%m-%d %HH") for key in data.keys()])

    # 检查是否有至少两个时间戳以进行比较
    if len(timestamps) < 2:
        return False

    # 检查所有时间戳是否都在每周的同一天同一时间
    day_of_week = timestamps[0].weekday()
    hour_of_day = timestamps[0].hour

    for ts in timestamps:
        if ts.weekday() != day_of_week or ts.hour != hour_of_day:
            return False

    # 检查相邻时间戳之间的间隔是否为整数个星期
    for i in range(len(timestamps) - 1):
        diff = timestamps[i + 1] - timestamps[i]
        if diff.days % 7 != 0:  # 确保间隔正好是一个整周
            return False

    return True


# 判断是不是日度
def is_within_35_days_before_date(data, target_date_str):
    # 解析目标日期
    target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d")

    # 计算目标日期的前35天日期
    start_date = target_date - datetime.timedelta(days=35)

    # 检查每个时间戳是否在指定范围内
    within_range = {}
    for key, value in data.items():
        timestamp = datetime.datetime.strptime(key, "%Y-%m-%d %HH")
        if start_date <= timestamp <= target_date:
            within_range[key] = value

    if len(data) == len(within_range):
        return True
    return False


# 判断当天是否是一些特殊的日期(先处理2025年度)
def check_special_date(target_date=None):
    special_dates_2025 = {
        (2, 14): '情人节',
        (4, 4): '清明节',
        (4, 5): '清明节',
        (4, 6): '清明节',
        (5, 1): '五一',
        (5, 2): '五一',
        (5, 3): '五一',
        (5, 4): '五一',
        (5, 5): '立夏',
        (5, 31): '端午节',
        (6, 1): '端午节',
        (6, 2): '端午节',
        (8, 29): '七夕',
        (8, 7): '立秋',
        (10, 1): '十一',
        (10, 2): '十一',
        (10, 3): '十一',
        (10, 4): '十一',
        (10, 5): '十一',
        (10, 7): '十一',
        (10, 8): '十一',
        (10, 6): '中秋节',
        (10, 29): '重阳节',
        (11, 7): '立冬',
        (11, 27): '感恩节',
        (12, 24): '平安夜',
        (12, 25): '圣诞节'
    }

    if target_date is None:
        target_date = datetime.date.today()

    if target_date.year != 2025:
        return None
    return special_dates_2025.get((target_date.month, target_date.day))


def get_shop_name(src_dict, shop_id):
    shop_name = ""
    for region_name in src_dict.keys():
        regions = src_dict.get(region_name, [])
        for region in regions:
            for store_name in region.get("所属门店", {}):
                if region["所属门店"][store_name] == shop_id:
                    shop_name = store_name
                    break
    return shop_name


def convert_types(d):
    converted_dict = {}
    for key, value in d.items():
        if isinstance(value, np.float64):
            converted_dict[key] = float(value)
        elif isinstance(value, Decimal):
            converted_dict[key] = float(value)
        else:
            # 如果不是需要特殊处理的类型，则直接保留原值
            converted_dict[key] = value
    return converted_dict
