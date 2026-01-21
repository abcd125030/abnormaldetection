import datetime
import json
import sys
import traceback

import pandas as pd
import requests
import time
from loguru import logger


class FeishuRobot:
    def __init__(self):
        self.app_id = 'cli_a5bdaa4d1770500c'
        self.app_secret = 't4d17VtkqJHce1mbyD2ihgr4wy1vcZmS'
        self.url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/'
        self.receive_id_test = ''
        self.open_id_test = ''
        self.user_id_test = ''
        self.header = None
        self.header_last_fetch = None

    def get_header(self):
        '''
        鉴权并获取飞书请求头。
        注:有效期2个小时
        https://open.feishu.cn/document/server-docs/authentication-management/access-token/tenant_access_token
        '''
        time.sleep(1)
        params = {"app_id": self.app_id,
                  "app_secret": self.app_secret}
        response = requests.post(self.url, params=params).json()
        if response['code'] == 0:
            access_token = response["tenant_access_token"]
            logger.bind(decorChaGee=True).info(f"获取api凭证成功:{access_token}")
        else:
            logger.bind(decorChaGee=True).info(f"获取api凭证失败:{response['msg']}")

        header = {"content-type": "application/json",
                  "Authorization": "Bearer " + access_token}
        return header

    def check_header(self):
        '''
        检查请求头是否过期
        '''
        if self.header_last_fetch is None:
            self.header = self.get_header()
            self.header_last_fetch = time.time()
        else:
            if time.time() - self.header_last_fetch > 7000:
                self.header = self.get_header()
                self.header_last_fetch = time.time()

    def get_user_id(self, user_id_type='user_id', in_type='emails', user_email_list=[]):
        self.check_header()
        url = f"https://open.feishu.cn/open-apis/contact/v3/users/batch_get_id?user_id_type={user_id_type}"
        df = None
        email_list = [user_email_list[i * 50:50 + i * 50] for i in range(0, len(user_email_list) // 50 + 1)]
        for in_list_ in email_list:
            payload = {in_type: in_list_}
            response = requests.post(url, headers=self.header, data=json.dumps(payload)).json()
            if response['code'] == 0:
                df_tmp = pd.DataFrame(response['data']['user_list'])
                df = pd.concat([df, df_tmp], axis=0)
                logger.bind(decorChaGee=True).info('成功:获取用户id')
            else:
                logger.bind(decorChaGee=True).info(f"失败:获取用户id{response}")
        return df.reset_index(drop=True)

    def post_robot(self, template_id, params: dict, receive_id_type='chat_id', receive_id=None):
        '''
        消息卡片推送
        # https://open.feishu.cn/tool/cardbuilder?templateId=ctp_AAyOO2yL9quP
        '''
        self.check_header()
        url = f'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}'

        if receive_id is None:
            receive_id = self.receive_id_test
            if receive_id_type == 'open_id':
                receive_id = self.open_id_test
            if receive_id_type == 'user_id':
                receive_id = self.user_id_test

        body = {
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": '{"type":"template","data":{"template_id":' + json.dumps(
                template_id) + ',"template_variable":' + json.dumps(params) + '}}'
        }
        response = requests.post(headers=self.header, url=url, data=json.dumps(body)).json()
        if response['code'] == 0:
            logger.bind(decorChaGee=True).info('发送成功')
        else:
            logger.bind(decorChaGee=True).info(f"{response}")
            return response

    def send_warning(self, receive_id=None, content=None):
        '''
        通过飞书机器人给自己推送告警信息：时间&路径&日志
        注：默认推送告警内容
        '''
        if content is None:
            content = traceback.format_exc()

        logger.bind(decorChaGee=True).info(f"send_warning:{content}")
        self.post_robot(template_id='ctp_AA1PEbOcIryZ'
                        , receive_id_type='user_id'
                        , receive_id=receive_id
                        , params={'warning_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                , 'warning_road': sys.path[0]
                , 'warning_content': content})