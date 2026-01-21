from loguru import logger
import warnings
import pandas as pd
import numpy as np
from config.conf import convert_types
from config.myOdps import PYODPS
warnings.filterwarnings('ignore')


def init():
    # 避免科学记数法
    np.set_printoptions(suppress=True)
    pd.reset_option("display.float_format")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 100)
    # plt.rcParams['font.sans-serif'] = ['SimHei']


# 获取每个大区下面所有的子公司以及子公司的名称和Id
# 因为子公司下属的门店有新开业的也有关店的，所以需要根据日期来获取最新的数据
def read_stores_data_yushitai(region_dict, date):
    result_dict = dict()
    # 昨天日期
    yesterday_date = pd.to_datetime(date) - pd.Timedelta(days=1)
    yesterday_date_str = yesterday_date.strftime('%Y%m%d')
    init()
    odps = PYODPS()
    store_cups_sql = f'''
                        SELECT 
                            name_full,
                            store_code,
                            company_name_full
                        FROM 
                            chagee_uat.dim_store_manage_emp_info_df as t0
                        WHERE 
                            not(t0.code='81000501' or t0.name_full in ('港澳办事处','港澳子公司')) 
                            and t0.company_name_full not in ('港澳办事处','港澳子公司')
                            and t0.level ='15'
                            and t0.store_valid_flag =1
                            and t0.store_name NOT REGEXP '样板|测试|备用|总部1|演示|集团公司fcc霸王学堂|x-机动队门店|管理有限公司|作废|未使用'
                            and t0.pt = {yesterday_date_str};
    '''
    # 计时开始
    logger.bind(decorChaGee=True).info(f'read_stores_data_yushitai--开始查询...')
    start_time = pd.Timestamp.now()
    df = odps.select_table(store_cups_sql)
    end_time = pd.Timestamp.now()
    logger.bind(decorChaGee=True).info(f'read_stores_data_yushitai--查询耗时：{end_time - start_time}')

    # 遍历DataFrame中的每一行数据
    for index, row in df.iterrows():
        # 获取当前行的子公司名称和对应的区域
        subsidiary_name = row['company_name_full']
        region = region_dict.get(subsidiary_name)
        if region:
            # 准备所属门店信息
            store_info = {row['name_full']: row['store_code']}

            # 如果该区域还没有在结果字典中出现，则初始化之
            if region not in result_dict:
                result_dict[region] = []

            # 查找是否已经存在相同子公司的记录
            subsidiary_exists = False
            for subsidiary in result_dict[region]:
                if subsidiary['子公司名称'] == subsidiary_name:
                    # 如果找到了相同的子公司，则直接添加或更新其门店信息
                    subsidiary['所属门店'].update(store_info)
                    subsidiary_exists = True
                    break

            # 如果没有找到相同的子公司，则新建一条记录
            if not subsidiary_exists:
                new_subsidiary_entry = {
                    "子公司名称": subsidiary_name,
                    "所属门店": store_info
                }
                result_dict[region].append(new_subsidiary_entry)

    return result_dict


# 将所有的store_ids的信息一次性取出，然后再做处理
def get_all_infos_ids(store_ids, date):
    result_dict_list = list()
    # 昨天日期
    yesterday_date = pd.to_datetime(date) - pd.Timedelta(days=1)
    yesterday_date_str = yesterday_date.strftime('%Y%m%d')
    # 前天日期
    the_day_before_yesterday_date = pd.to_datetime(date) - pd.Timedelta(days=2)
    the_day_before_yesterday_date_str = the_day_before_yesterday_date.strftime('%Y%m%d')

    init()
    odps = PYODPS()
    # 将store_ids列表转换为逗号分隔的字符串
    store_ids_str = ', '.join(map(str, store_ids))
    store_gmv_sql = f'''
        SELECT 
            t1.store_code AS store_code,
            COALESCE(t3.material_cost, 0) AS material_cost,
            COALESCE(t4.all_cnt, 0) AS all_cnt,
            COALESCE(t2.total_frmloss_price, 0) AS total_frmloss_price,
            COALESCE(t4.total_salary, 0) AS total_salary,
            COALESCE(t4.ave_salary, 0) AS ave_salary,
            COALESCE(ABS(t5.consumables_expenses), 0) AS consumables_expenses,
            COALESCE(ABS(t5.maintenance_expenses), 0) AS maintenance_expenses,
            COALESCE(ABS(t5.training_fees), 0) AS training_fees,
            COALESCE(ABS(t5.travel_expenses), 0) AS travel_expenses,
            COALESCE(ABS(t5.work_clothing_cost), 0) AS work_clothing_cost,
            COALESCE(ABS(t5.rental_cost), 0) AS rental_cost,
            COALESCE(ABS(t5.daily_expenses), 0) AS other_cost,
            COALESCE(ABS(t1.total_gmv), 0) AS total_gmv,
            (
                COALESCE(t1.total_nra, 0) - 
                COALESCE(t4.total_salary, 0) - 
                (COALESCE(t3.material_cost, 0) + COALESCE(t3.supply_chain_delivery_cost, 0) + COALESCE(t2.total_frmloss_price, 0)) + 
                COALESCE(t5.rental_cost, 0) + 
                COALESCE(t5.brand_expenses, 0) + 
                COALESCE(t5.marketing_expenses, 0) + 
                COALESCE(t5.daily_expenses, 0) + 
                COALESCE(t5.personnel_management, 0)
            ) AS ebitda,
            COALESCE(
                ABS(
                    CASE 
                        WHEN t1.total_gmv = 0 THEN 0 -- 防止除以零
                        ELSE (
                            (
                                COALESCE(t1.total_nra, 0) - 
                                COALESCE(t4.total_salary, 0) - 
                                (COALESCE(t3.material_cost, 0) + COALESCE(t3.supply_chain_delivery_cost, 0) + COALESCE(t2.total_frmloss_price, 0)) + 
                                COALESCE(t5.rental_cost, 0) + 
                                COALESCE(t5.brand_expenses, 0) + 
                                COALESCE(t5.marketing_expenses, 0) + 
                                COALESCE(t5.daily_expenses, 0) + 
                                COALESCE(t5.personnel_management, 0)
                            ) / NULLIF(t1.total_gmv, 0)
                        ) -- 使用NULLIF防止除以零
                    END
                ), 
                0
            ) AS ebitda_to_cost_ratio,
            COALESCE(t1.total_nra_gmv_ratio, 0) AS total_nra_gmv_ratio,
            (
                ABS(
                    COALESCE(t3.material_cost, 0) + 
                    COALESCE(t3.supply_chain_delivery_cost, 0) + 
                    COALESCE(t2.total_frmloss_price, 0)
                )
            ) AS good_to_cost,
            COALESCE(
                ABS(
                    CASE 
                        WHEN t1.total_gmv = 0 THEN 0 -- 防止除以零
                        ELSE (
                            (
                                COALESCE(t3.material_cost, 0) +
                                COALESCE(t3.supply_chain_delivery_cost, 0) + 
                                COALESCE(t2.total_frmloss_price, 0)
                            ) / NULLIF(t1.total_gmv, 0)
                        ) -- 使用NULLIF防止除以零
                    END
                ), 
                0
            ) AS good_to_cost_ratio,
            COALESCE(t4.gmv_d, 0) AS gmv_d, 
            COALESCE(t4.people_to_sales_ratio, 0) AS people_to_sales_ratio,
            COALESCE(
                ABS(
                    CASE 
                        WHEN t1.total_gmv = 0 THEN 0 -- 防止除以零
                        ELSE (t5.rental_cost / NULLIF(t1.total_gmv, 0)) -- 使用NULLIF防止除以零
                    END
                ), 
                0
            ) AS rental_cost_to_gmv_ratio,
            COALESCE(
                ABS(
                    CASE 
                        WHEN t1.total_gmv = 0 THEN 0 -- 防止除以零
                        ELSE (t5.daily_expenses / NULLIF(t1.total_gmv, 0)) -- 使用NULLIF防止除以零
                    END
                ), 
                0
            ) AS other_cost_to_gmv_ratio,
            COALESCE(t1.total_nra, 0) AS total_nra
        FROM 
            (
                SELECT
                    store_code,
                    SUM(gmv) AS total_gmv,
                    SUM(nra) AS total_nra,
                    (SUM(nra) / NULLIF(SUM(gmv), 0)) AS total_nra_gmv_ratio
                FROM 
                    chagee_ads_prod.ads_store_channel_goods_summary_di
                WHERE 
                    pt = {yesterday_date_str}
                    AND store_code IN ({store_ids_str})
                GROUP BY 
                    store_code
            ) AS t1
        LEFT JOIN 
            (
                SELECT 
                    store_code,
                    SUM(frmloss_price) AS total_frmloss_price
                FROM 
                    chagee_uat.ads_inventory_produce_frmloss_di
                WHERE 
                    pt = {yesterday_date_str}
                    AND store_code IN ({store_ids_str})
                GROUP BY 
                    store_code
            ) AS t2 ON t1.store_code = t2.store_code
        LEFT JOIN 
            (
                SELECT 
                    store_code,
                    other_ingredients_cost,
                    agricultural_products_cost,
                    packaging_materials_cost,
                    peripheral_customization_cost,
                    dairy_products_cost,
                    jam_cost,
                    tea_cost,
                    milk_cost,
                    baking_cost,
                    supply_chain_delivery_cost,
                    (
                        COALESCE(other_ingredients_cost, 0) + 
                        COALESCE(agricultural_products_cost, 0) + 
                        COALESCE(packaging_materials_cost, 0) + 
                        COALESCE(peripheral_customization_cost, 0) + 
                        COALESCE(dairy_products_cost, 0) + 
                        COALESCE(jam_cost, 0) + 
                        COALESCE(tea_cost, 0) + 
                        COALESCE(milk_cost, 0) + 
                        COALESCE(baking_cost, 0)
                    ) AS material_cost
                FROM 
                    chagee_ads_prod.ads_franchise_material_cost_di
                WHERE 
                    pt = {yesterday_date_str}
                    AND store_code IN ({store_ids_str})
            ) AS t3 ON t1.store_code = t3.store_code
        LEFT JOIN 
            (
                SELECT 
                    store_code,
                    total_salary,
                    gmv_d,
                    all_cnt,
                    total_salary / all_cnt AS ave_salary,
                    total_salary / gmv_d AS people_to_sales_ratio
                FROM 
                    chagee_uat.ads_store_piece_rate_salary_di
                WHERE 
                    pt = {the_day_before_yesterday_date_str}
                    AND store_code IN ({store_ids_str})
            ) AS t4 ON t1.store_code = t4.store_code
        LEFT JOIN 
            (
                SELECT 
                    store_code_bawang,
                    consumables_expenses,
                    maintenance_expenses,
                    work_clothing_cost,
                    training_fees,
                    travel_expenses,
                    rental_cost,
                    daily_expenses,
                    brand_expenses,
                    marketing_expenses,
                    personnel_management
                FROM 
                    chagee_dwd_prod.dwd_finance_ue_detail_flat_avg_di
                WHERE 
                    pt = {yesterday_date_str}
                    AND store_code_bawang IN ({store_ids_str})
            ) AS t5 ON t1.store_code = t5.store_code_bawang;
    '''
    # 计时开始
    logger.bind(decorChaGee=True).info(f'开始查询...')
    start_time = pd.Timestamp.now()
    df = odps.select_table(store_gmv_sql)
    if not df.empty:
        # 遍历DataFrame的每一行
        for index, row in df.iterrows():
            # 将每一行转换为字典并添加到列表中
            single_shop_dict = dict()
            for column in df.columns:
                if column == "store_code":
                    single_shop_dict[column] = str(row[column])
                if column in ['material_cost', 'total_frmloss_price', 'ave_salary',
                              'consumables_expenses', 'maintenance_expenses',
                              'other_cost', 'personnel_management',
                              'training_fees', 'travel_expenses',
                              'work_clothing_cost', 'rental_cost', 'other_cost', 'good_to_cost',
                              'ebitda', 'total_gmv', 'total_nra'
                              ]:
                    if row[column]:
                        single_shop_dict[column] = round(row[column], 2)
                    else:
                        single_shop_dict[column] = 0.0
                if column in ['total_salary', 'gmv_d']:
                    if row[column]:
                        single_shop_dict[column] = round(float(row[column]), 2)
                    else:
                        single_shop_dict[column] = 0.0
                if column in ['all_cnt']:
                    if row[column]:
                        single_shop_dict[column] = int(row[column])
                    else:
                        single_shop_dict[column] = 0
                if column in ['ebitda_to_cost_ratio', 'total_nra_gmv_ratio',
                              'good_to_cost_ratio', 'people_to_sales_ratio',
                              'rental_cost_to_gmv_ratio', 'other_cost_to_gmv_ratio'
                              ]:
                    if row[column]:
                        single_shop_dict[column] = round(row[column], 4)
                    else:
                        single_shop_dict[column] = 0.0

            result_dict_list.append(convert_types(single_shop_dict))

    end_time = pd.Timestamp.now()
    logger.bind(decorChaGee=True).info(f'查询耗时：{end_time - start_time}')
    return result_dict_list
