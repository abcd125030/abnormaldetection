# -*- coding: utf-8 -*-
# @Time : 2024/11/12 14:43
# @File : start.py
# @Author : lisomgming
import os
from multiprocessing import cpu_count
from loguru import logger
from sanic import Sanic
import traceback
import schedule
import time
from multiprocessing import Process
from api import api
from api.yushitai.forecast import forecast
from config import conf


app = Sanic(__name__)
app.blueprint(api)


@app.listener('before_server_start')
async def setup_db(app, loop):
    app.config.REQUEST_MAX_SIZE = 409600000
    logger.bind(decorChaGee=True).info("==========setup_oss==========")

@app.listener('after_server_start')
async def notify_server_started(app, loop):
    logger.bind(decorChaGee=True).info("Server successfully started!")


@app.listener('before_server_stop')
async def notify_server_stopping(app, loop):
    logger.bind(decorChaGee=True).info("Server shutting down!")


@app.listener('after_server_stop')
async def close_connection(app, loop):
    logger.bind(decorChaGee=True).info("==========close_oss==========")


def run_sanic():
    """启动 Sanic 应用"""
    env_type = os.getenv("ENV", "PROD")
    logger.bind(decorChaGee=True).info("env_type: " + str(env_type))
    workers_num = cpu_count()
    if env_type in ['dev', 'test', 'uat']:
        workers_num = 1
    logger.bind(decorChaGee=True).info("workers_num: " + str(workers_num))
    app.run(
        host="0.0.0.0",
        port=8080,
        workers=workers_num,
        debug=False,
        access_log=False,
    )


def run_schedule():
    """在测试环境和生成环境下启动定时任务"""
    env_type = os.getenv("ENV", "PROD")
    if env_type == "PROD" or env_type == "prod":
        schedule_time = "11:10"
        logger.bind(decorChaGee=True).info("run_schedule-begin-" + "env_type: " + env_type + "-schedule_time: " + schedule_time)
        schedule.every().day.at(schedule_time).do(forecast)
        while True:
            schedule.run_pending()
            time.sleep(1)
    else:
        logger.bind(decorChaGee=True).info("run_schedule-not-begin-" + "env_type: " + env_type)


if __name__ == "__main__":
    sanic_process = None
    schedule_process = None

    try:
        # 启动 Sanic 应用的进程
        sanic_process = Process(target=run_sanic)
        sanic_process.start()
        logger.bind(decorChaGee=True).info("Sanic server started.")
        # 启动定时任务的进程
        schedule_process = Process(target=run_schedule)
        schedule_process.start()
        logger.bind(decorChaGee=True).info("Schedule process started.")
        # 等待两个进程结束
        sanic_process.join()
        schedule_process.join()
    except Exception as e:
        logger.bind(decorChaGee=True).warning(traceback.format_exc() + "--" + str(e))
        logger.bind(decorChaGee=True).info("run error, processes terminated.")
        if sanic_process:
            sanic_process.terminate()
        if schedule_process:
            schedule_process.terminate()
        logger.bind(decorChaGee=True).info("run error, processes terminated.")

