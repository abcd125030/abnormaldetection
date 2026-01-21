import json

from loguru import logger

from config.conf import find_om_v3_result_all, getcontent


def deal_with_last_week(week_info_new_dict):
    all_last_week_json = list()
    for key in week_info_new_dict.keys():
        if key.find("上周") >= 0:
            time_day_day = key.split("+")[2]
            last_week_json = dict()
            last_week_json[time_day_day] = week_info_new_dict[key]
            all_last_week_json.append(last_week_json)

    if len(all_last_week_json) > 0:
        dir = "../data/核心指标/周_核心指标/上周"
        with open(dir + "/" + "data.json", 'w') as f:
            json.dump(all_last_week_json, f)

def deal_with_this_week(week_info_new_dict):
    all_today_json = list()
    for key in week_info_new_dict.keys():
        if key.find("当周") >= 0:
            time_day_day = key.split("+")[2] + "+" + key.split("+")[3]
            today_json = dict()
            today_json[time_day_day] = week_info_new_dict[key]
            all_today_json.append(today_json)

    # if len(all_today_json) > 0:
    #     dir = "../data/核心指标/周_核心指标/当周"
    #     for hour in range(0, 24):
    #         all_today_hour_json = list()
    #         if not os.path.exists(dir + "/" + str(hour) + "点"):
    #             os.mkdir(dir + "/" + str(hour) + "点")
    #         for today_json in all_today_json:
    #             for today_key in today_json.keys():
    #                 today_key_hour = int(today_key.split("+")[1].split("-")[0])
    #                 if hour == today_key_hour:
    #                     all_today_hour_json.append(today_json)
    #         with open(dir + "/" + str(hour) + "点" + "/" + "data.json", 'w') as f:
    #             json.dump(all_today_hour_json, f)


if __name__ == "__main__":
    logger.bind(decorChaGee=True).info('核心指标-数据库读取开始')
    results = find_om_v3_result_all("核心指标",0, 24)
    logger.bind(decorChaGee=True).info('核心指标-数据库读取完毕')
    if len(results) > 0:
        logger.bind(decorChaGee=True).info('核心指标-数据处理开始')
        all_time_last_week_hours = list()
        all_time_this_week_hours = list()
        week_info_new_dict = dict()
        for result in results:
            process_time_hour = result['process_time'].strftime('%Y-%m-%d %H-%M-%S')
            print(process_time_hour)
            process_time = process_time_hour.split(" ")[0]
            time_last_week_hour = process_time
            hour = process_time_hour.split(" ")[1].split(":")[0]
            time_this_week_hour = process_time + "+" + hour
            if time_last_week_hour not in all_time_last_week_hours:
                all_time_last_week_hours.append(time_last_week_hour)
                week_info_new_dict["上周" + "+" + "核心指标" + "+" + time_last_week_hour] = getcontent(
                    result['datas_json'], "上周", "核心指标", True)
            if time_this_week_hour not in all_time_this_week_hours:
                all_time_this_week_hours.append(time_this_week_hour)
                week_info_new_dict["当周" + "+" + "核心指标" + "+" + time_this_week_hour] = getcontent(
                    result['datas_json'], "当周", "核心指标", True)


        # 开始处理周核心指标
        logger.bind(decorChaGee=True).info('核心指标-处理周核心指标开始')
        deal_with_last_week(week_info_new_dict)
        deal_with_this_week(week_info_new_dict)
        logger.bind(decorChaGee=True).info('核心指标-处理周核心指标结束')


        logger.bind(decorChaGee=True).info('核心指标-数据处理结束')