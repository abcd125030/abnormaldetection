# -*- coding: utf-8 -*-
# @Time : 2024/11/12 14:38
# @File : bp.py
# @Author : lisongming
import datetime
import os
import time
import traceback
import uuid
from multiprocessing import cpu_count

from loguru import logger
from sanic import Blueprint
from sanic.response import json
import json as normal_json

from api.script.algorithm import my_algorithm
from config.conf import get_normal_interval_from_mysql_adapter

views_bp = Blueprint('views_bp')


@views_bp.route("/version", methods=['Get'])
async def version(request):
    try:
        start_time = time.time()
        logger.bind(decorChaGee=True).info("=================================================")
        logger.bind(decorChaGee=True).info("get version 2.0")
        env_type = os.getenv("ENV", "PROD")
        logger.bind(decorChaGee=True).info("abnormaldetection-" + env_type)
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')
        logger.bind(decorChaGee=True).info("=================================================")
        return_json = json({"version": "abnormaldetection-2.0" + str(time.time() - start_time), "cpu_count()": str(cpu_count()), "today_date": today_date, "ENV_TYPE": env_type}, 200)
        logger.bind(decorChaGee=True).info("abnormaldetection-2.0-" + str(time.time() - start_time))
        logger.bind(decorChaGee=True).info("=================================================")
        return return_json
    except Exception as e:
        logger.bind(decorChaGee=True).error(traceback.format_exc() + ' ' + str(e))
        return json({"status": "Error", "data": {"SchemeList": []}}, 500)


@views_bp.route("/check_abnormal", methods=['POST'])
async def checkabnormal(request):
    try:
        logger.bind(decorChaGee=True).info("=================================================")
        check_abnormal_label_id = str(uuid.uuid4()).replace('-', '')
        logger.bind(decorChaGee=True).info(normal_json.dumps(request.json, ensure_ascii=False)
                                           + "| " + check_abnormal_label_id + " ")
        outlier_result, outlier_result_time = my_algorithm(request.json)
        logger.bind(decorChaGee=True).info("check_abnormal 2.0")
        logger.bind(decorChaGee=True).info("=================================================")

        status = "Success"
        index = 200
        if len(outlier_result) == 1 and outlier_result[0]['status'] == -1:
            status = "Error"
            index = 500
        # if outlier_result['status'] == -1:
        #     status = "Error"
        #     index = 500
        output_dic = dict()
        output_dic["status"] = status
        output_dic["result"] = outlier_result
        output_dic["test_time"] = outlier_result_time
        return_json = json(output_dic, index)
        logger.bind(decorHaier=True).info(normal_json.dumps(output_dic, ensure_ascii=False))
        logger.bind(decorChaGee=True).info("=================================================")
        return return_json
    except Exception as e:
        logger.bind(decorChaGee=True).error(traceback.format_exc() + ' ' + str(e))
        return json({"status": "Error", "data": {"SchemeList": []}}, 500)
