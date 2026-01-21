from config.conf import get_shop_name


def deal_with_shop_dict(country_name, region_name, company_name, store_info_dict, shop_dict_list):
    # 避免科学记数法
    column_names = ['ebitda', 'total_gmv', 'ebitda_to_cost_ratio', 'total_nra_gmv_ratio',
                    'good_to_cost_ratio', 'people_to_sales_ratio', 'rental_cost_to_gmv_ratio',
                    'other_cost_to_gmv_ratio', 'total_nra', 'material_cost',
                    'all_cnt', 'total_frmloss_price', 'ave_salary',
                    'consumables_expenses', 'maintenance_expenses',
                    'training_fees', 'travel_expenses', 'work_clothing_cost']
    result_dicts = list()
    for shop_dict in shop_dict_list:
        result_dict = dict()
        result_dict['country_name'] = country_name
        result_dict['region_name'] = region_name
        result_dict['company_name'] = company_name
        result_dict['shop_name'] = get_shop_name(store_info_dict, str(shop_dict['store_code']))
        # print(result_dict['shop_name'])
        for column_name in column_names:
            result_dict[column_name] = shop_dict[column_name]
        result_dicts.append(result_dict)
    return result_dicts


def deal_with_regin_company_dict(country_name, region_name, company_name, shop_dict_list):
    #
    result_dict = dict()
    result_dict['region_name'] = region_name
    result_dict['company_name_name'] = company_name
    #
    result_dict = dict()
    result_dict['country_name'] = country_name
    result_dict['region_name'] = region_name
    result_dict['company_name'] = company_name
    result_dict['shop_name'] = ""
    # 第一页的内容
    all_ebitda = 0.0
    all_total_gmv = 0.0
    all_total_nra = 0.0
    all_good_to_cost = 0.0
    # 累加T-2日GMV
    all_gmv_d = 0.0
    all_rental_cost = 0.0
    all_other_cost = 0.0
    # 第二页的内容
    all_material_cost = 0.0
    all_people_cnt = 0.0
    all_total_frmloss_price = 0.0
    all_salary = 0.0
    all_consumables_expenses = 0.0
    all_maintenance_expenses = 0.0
    all_training_fees = 0.0
    all_travel_expenses = 0.0
    all_work_clothing_cost = 0.0
    # 开始累加
    for shop_dict in shop_dict_list:
        all_ebitda += shop_dict['ebitda']
        all_total_gmv += shop_dict['total_gmv']
        all_total_nra += shop_dict['total_nra']
        all_good_to_cost += shop_dict['good_to_cost']
        # 累加T-2日GMV
        all_gmv_d += shop_dict['gmv_d']
        all_rental_cost += shop_dict['rental_cost']
        all_other_cost += shop_dict['other_cost']
        #
        all_material_cost += shop_dict['material_cost']
        all_people_cnt += shop_dict['all_cnt']
        all_total_frmloss_price += shop_dict['total_frmloss_price']
        all_salary += shop_dict['total_salary']
        all_consumables_expenses += shop_dict['consumables_expenses']
        all_maintenance_expenses += shop_dict['maintenance_expenses']
        all_training_fees += shop_dict['training_fees']
        all_travel_expenses += shop_dict['travel_expenses']
        all_work_clothing_cost += shop_dict['work_clothing_cost']

    all_ave_salary = 0.0
    if all_people_cnt > 0:
        all_ave_salary = round(all_salary / all_people_cnt, 2)

    all_total_ebitda_gmv_ratio = 0.0
    if all_total_gmv > 0.0:
        all_total_ebitda_gmv_ratio = round(all_ebitda / all_total_gmv, 4)

    all_total_nra_gmv_ratio = 0.0
    if all_total_gmv > 0.0:
        all_total_nra_gmv_ratio = round(all_total_nra / all_total_gmv, 4)

    all_store_good_to_cost_ratio = 0.0
    if all_total_gmv > 0.0:
        all_store_good_to_cost_ratio = round(all_good_to_cost / all_total_gmv, 4)

    all_store_people_to_sales_ratio = 0.0
    if all_gmv_d > 0.0:
        all_store_people_to_sales_ratio = round(all_salary / all_gmv_d, 4)

    all_store_rental_cost_to_gmv_ratio = 0.0
    if all_total_gmv > 0.0:
        all_store_rental_cost_to_gmv_ratio = round(all_rental_cost / all_total_gmv, 4)

    all_store_other_cost_to_gmv_ratio = 0.0
    if all_total_gmv > 0.0:
        all_store_other_cost_to_gmv_ratio = round(all_other_cost / all_total_gmv, 4)

    result_dict['ebitda'] = all_ebitda
    result_dict['total_gmv'] = all_total_gmv
    result_dict['ebitda_to_cost_ratio'] = all_total_ebitda_gmv_ratio
    result_dict['total_nra_gmv_ratio'] = all_total_nra_gmv_ratio
    result_dict['good_to_cost_ratio'] = all_store_good_to_cost_ratio
    result_dict['people_to_sales_ratio'] = all_store_people_to_sales_ratio
    result_dict['rental_cost_to_gmv_ratio'] = all_store_rental_cost_to_gmv_ratio
    result_dict['other_cost_to_gmv_ratio'] = all_store_other_cost_to_gmv_ratio
    result_dict['total_nra'] = all_total_nra
    result_dict['material_cost'] = all_material_cost
    result_dict['all_cnt'] = all_people_cnt
    result_dict['total_frmloss_price'] = all_total_frmloss_price
    result_dict['ave_salary'] = all_ave_salary
    result_dict['consumables_expenses'] = all_consumables_expenses
    result_dict['maintenance_expenses'] = all_maintenance_expenses
    result_dict['training_fees'] = all_training_fees
    result_dict['travel_expenses'] = all_travel_expenses
    result_dict['work_clothing_cost'] = all_work_clothing_cost

    return result_dict
