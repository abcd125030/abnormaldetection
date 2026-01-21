import datetime
import os
import traceback
import pandas as pd
from loguru import logger

from api.yushitai import path_timeforecasting
from api.yushitai.dataExtraction import read_stores_data_yushitai, get_all_infos_ids
from api.yushitai.dealwith import deal_with_shop_dict, deal_with_regin_company_dict
from config.FeishuRobot import FeishuRobot
from config.conf import database_config
from config.database_pymysql import DbSqlalchemy
from sqlalchemy import TIMESTAMP


def forecast():
    try:
        send_feishu('统计驭事台数据开始')
        # 获取系统当前日期
        forecast_day = datetime.datetime.today() - datetime.timedelta(days=0)
        date = forecast_day.strftime('%Y-%m-%d')
        # 获取全国所有的大区名称
        df = pd.read_csv(os.path.dirname(path_timeforecasting) + "/data/霸王茶姬现有的大区以及下属子公司.csv")
        region_dict = dict(zip(df.company_name, df.region))
        # print(region_dict)
        # 读取目前所有在运营的门店
        store_info_dict = dict()
        try:
            store_info_dict = read_stores_data_yushitai(region_dict, date)
        except Exception as e:
            logger.bind(decorChaGee=True).info(f'获取目前所有在运营的门店失败。error：{e}')
            send_feishu('获取目前所有在运营的门店失败。error：{e}')
        # print(store_info_dict)

        if len(store_info_dict) == 0:
            logger.bind(decorChaGee=True).info('未获取到目前所有在运营的门店, 程序终止')
            send_feishu('未获取到目前所有在运营的门店, 程序终止')
            return
        logger.bind(decorChaGee=True).info('成功获取到目前所有在运营的门店')
        send_feishu('成功获取到目前所有在运营的门店')

        # 获取每个门店的运行信息并计算总体的GMV和NRA
        all_result_dict_list = list()
        #
        all_country_dict_list = list()
        for region in store_info_dict.keys():
            # print(region)
            all_region_dict_list = list()
            # if region == "华北大区":
            for sub_company in store_info_dict[region]:
                # print(sub_company['子公司名称'])
                all_sub_company_dict_list = list()
                # if sub_company['子公司名称'] == "吉林办事处":
                all_sub_company_shop_ids = list()
                for sub_shop_name in sub_company['所属门店'].keys():
                    # print(sub_shop_name)
                    shop_id = sub_company['所属门店'][sub_shop_name]
                    all_sub_company_shop_ids.append(shop_id)
                if len(all_sub_company_shop_ids) > 0:
                    # print(len(all_sub_company_shop_ids))
                    # 获取每个门店的运行信息并计算GMV等数据
                    try:
                        shop_dict_list = get_all_infos_ids(all_sub_company_shop_ids, date)
                        all_sub_company_dict_list.extend(shop_dict_list)
                        all_region_dict_list.extend(shop_dict_list)
                        all_country_dict_list.extend(shop_dict_list)
                        shop_result_dicts = deal_with_shop_dict("中国大陆", region, sub_company['子公司名称'],
                                                                store_info_dict, shop_dict_list)
                        all_result_dict_list.extend(shop_result_dicts)
                    except Exception as e:
                        logger.bind(decorChaGee=True).error(f'获取门店GMV等信息失败。error：'
                                                            + traceback.format_exc() + ' ' + str(e))
                        send_feishu(f'获取门店GMV等信息失败。error：' + traceback.format_exc() + ' ' + str(e))
                if len(all_sub_company_dict_list):
                    regin_company_dict = deal_with_regin_company_dict("中国大陆", region,
                                                                      sub_company['子公司名称'], all_sub_company_dict_list)
                    all_result_dict_list.append(regin_company_dict)
            #
            if len(all_region_dict_list) > 0:
                regin_company_dict = deal_with_regin_company_dict("中国大陆", region,
                                                                  "", all_region_dict_list)
                all_result_dict_list.append(regin_company_dict)

        if len(all_country_dict_list) > 0:
            country_company_dict = deal_with_regin_company_dict("中国大陆", "",
                                                                "", all_country_dict_list)
            all_result_dict_list.append(country_company_dict)

        # print(len(all_result_dict_list))
        logger.bind(decorChaGee=True).info('将所有的门店的驭事台信息写入数据库中--开始')
        send_feishu('将所有的门店的驭事台信息写入数据库中--开始')
        if len(all_result_dict_list) > 0:
            host, port, username, password, database = database_config()
            db_sqlalchemy = DbSqlalchemy()
            db_sqlalchemy.dbSetup(host, port, username, password, database)
            yesterday_date = pd.to_datetime(date) - pd.Timedelta(days=1)
            yesterday_date_str = yesterday_date.strftime('%Y-%m-%d')
            all_input_mysql_data = list()
            for result_dict in all_result_dict_list:
                input_mysql_data = {
                    'process_time': [datetime.datetime.now()],
                    'country': result_dict['country_name'],
                    "date": yesterday_date_str,
                    "region": result_dict['region_name'],
                    "company": result_dict['company_name'],
                    "shop": result_dict['shop_name'],
                    "ebitda": str(round(result_dict['ebitda'], 2)),
                    "gmv": str(round(result_dict['total_gmv'], 2)),
                    "ebitda_to_gmv_ratio": str(round(result_dict['ebitda_to_cost_ratio'], 4)),
                    "nra_gmv_ratio": str(round(result_dict['total_nra_gmv_ratio'], 4)),
                    "good_gmv_ratio": str(round(result_dict['good_to_cost_ratio'], 4)),
                    "people_gmv_ratio": str(round(result_dict['people_to_sales_ratio'], 4)),
                    "rental_gmv_ratio": str(round(result_dict['rental_cost_to_gmv_ratio'], 4)),
                    "other_gmv_ratio": str(round(result_dict['other_cost_to_gmv_ratio'], 4)),
                    "nra": str(round(result_dict['total_nra'], 2)),
                    "material_cost": str(round(result_dict['material_cost'], 2)),
                    "all_cnt": str(round(result_dict['all_cnt'], 0)),
                    "total_frmloss_price": str(round(result_dict['total_frmloss_price'], 2)),
                    "ave_salary": str(round(result_dict['ave_salary'], 2)),
                    "consumables_expenses": str(round(result_dict['consumables_expenses'], 2)),
                    "maintenance_expenses": str(round(result_dict['maintenance_expenses'], 2)),
                    "training_fees": str(round(result_dict['training_fees'], 2)),
                    "travel_expenses": str(round(result_dict['travel_expenses'], 2)),
                    "work_clothing_cost": str(round(result_dict['work_clothing_cost'], 2)),
                }
                all_input_mysql_data.append(input_mysql_data)
            print(len(all_input_mysql_data))
            try:
                # 将DataFrame写入数据库
                df = pd.DataFrame(all_input_mysql_data)
                df.to_sql('om_v4_yushitai_detection_result', con=db_sqlalchemy.engine, index=False, if_exists='append',
                          dtype={'process_time': TIMESTAMP})
                logger.bind(decorHaier=True).info('驭事台信息写入sql成功')
                send_feishu('驭事台信息写入sql成功')
            except Exception as e:
                logger.bind(decorHaier=True).info('驭事台信息写入sql错误, ' + traceback.format_exc() + ' ' + str(e))
                send_feishu('驭事台信息写入sql错误, ' + traceback.format_exc() + ' ' + str(e))
        send_feishu('将所有的门店的驭事台信息写入数据库中--结束')
        logger.bind(decorChaGee=True).info('将所有的门店的驭事台信息写入数据库中--结束')
    except Exception as e:
        logger.bind(decorChaGee=True).error(traceback.format_exc() + ' ' + str(e))
    logger.bind(decorChaGee=True).info('统计驭事台数据结束')


# 飞书告警
def send_feishu(content):
    env_type = os.getenv("ENV", "PROD")
    robot = FeishuRobot()
    user_email_list_algo = ['lisongming@chagee.com']
    try:
        # 根据需要改自己的邮箱就可以
        user_ids = robot.get_user_id(user_email_list=user_email_list_algo)['user_id']
        for user_id in user_ids:
            logger.bind(decorChaGee=True).info(user_id)
            # 自定义内容推送
            robot.send_warning(receive_id=user_id, content=content + "-env_type-" + env_type)
    except Exception as e:
        logger.bind(decorChaGee=True).error(traceback.format_exc() + ' ' + str(e) + "-env_type-" + env_type)
        # 告警推送
        user_ids = robot.get_user_id(user_email_list=user_email_list_algo)['user_id']
        for user_id in user_ids:
            robot.send_warning(receive_id=user_id, content=str(e) + "-env_type-" + env_type)


# 检测双方的数据，找出差异
def check_differ():
    send_feishu('check_differ -- begin')
    try:
        error_list = pd.DataFrame({'period': ['当月'] * 5, 'target': ['新增门店'] * 5,
                                   'target_value': [100, 200, 300, 400, 500],
                                   'low_limit': [0, 0, 0, 0, 0], 'high_limit': [100, 200, 300, 400, 500],
                                   'abnormal': [1, 1, 1, 1, 1]})
        send_feishu_error_list(error_list)
    except Exception as e:
        logger.bind(decorChaGee=True).error(traceback.format_exc() + ' ' + str(e))
        # 告警推送
        send_feishu('check_differ -- error' + traceback.format_exc() + ' ' + str(e))
    send_feishu('check_differ -- end')


# 如果检测双方数据差异过大，则需要将告警信息推送到告警群中
def send_feishu_error_list(content):
    feishu = FeishuRobot()
    if not content.empty:
        # 'AAqBYf7CKDrHo'为事先建好的告警群Id
        feishu.post_robot(template_id='AAqBYf7CKDrHo', receive_id_type='chat_id',
                          params={'title': "茶姬驭事台数据对比预警",
                                  'warn_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  'warn_freq': '1天1次', 'warn_people': '李宋明', 'warn_status': '未处理',
                                  'data_url': '[明细数据](https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce)',
                                  'data_compare': content.to_dict(orient='records')})
    else:
        feishu.post_robot(template_id='AAqBYzPVX6cjr', receive_id_type='chat_id',
                          params={'title': "茶姬驭事台数据对比预警",
                                  'warn_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  'warn_freq': '1天1次', 'warn_people': '李宋明',
                                  'warn_status': '正常',
                                  'data_url': '[明细数据](https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce)'})


if __name__ == '__main__':
    # 每天统计
    forecast()
