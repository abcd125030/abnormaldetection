# 标准库导入
import sys
import time
from typing import Dict, List, Optional
# 第三方库导入
import pandas as pd
import requests
from loguru import logger
from requests_toolbelt import MultipartEncoder
from tenacity import retry, stop_after_attempt, wait_incrementing


class FeishuBase:
    """飞书API基类"""
    BASE_URL = 'https://open.feishu.cn/open-apis'

    def __init__(self):
        self.app_id = 'cli_a5bdaa4d1770500c'
        self.app_secret = 't4d17VtkqJHce1mbyD2ihgr4wy1vcZmS'
        self.header = None
        self.header_last_fetch = None

    def get_header(self) -> Dict:
        """鉴权并获取认证请求头,有效期2小时
        :return: 包含认证信息的header字典
        """
        time.sleep(1)
        params = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        try:
            response = requests.post(self.AUTH_URL, params=params).json()

            if response['code'] != 0:
                error_msg = f"code={response['code']}, msg={response['msg']}"
                logger.error(f"错误: [API凭证] {error_msg}")
                raise Exception(error_msg)

            access_token = response["tenant_access_token"]
            logger.info("操作成功: 获取API凭证")

            return {
                "content-type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
        except Exception as e:
            logger.error(f"错误: [API凭证] {str(e)}")
            raise

    def check_header(self):
        """检查并更新header
        """
        if self.header_last_fetch is None or time.time() - self.header_last_fetch > 7000:
            self.header = self.get_header()
            self.header_last_fetch = time.time()

    @retry(stop=stop_after_attempt(5), wait=wait_incrementing(start=10, increment=30), reraise=True)
    def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        """统一的请求处理方法
        :param method: HTTP请求方法
        :param url: 请求URL
        :param kwargs: 其他请求参数
        :return: API响应数据字典
        """
        self.check_header()
        # 获取真实的调用者，跳过所有装饰器
        frame = sys._getframe(1)
        while frame and frame.f_code.co_name in ('__call__', 'wrapper', 'wrapped_f', 'retry', 'retry_with_cleanup'):
            frame = frame.f_back
        caller = frame.f_code.co_name if frame else 'unknown'
        try:
            response = requests.request(method, url, headers=self.header, **kwargs).json()

            if response['code'] != 0:
                error_msg = f"caller:{caller}, code:{response['code']}, msg:{response.get('msg', 'Unknown error')}"
                raise Exception(error_msg)

            return response

        except Exception as e:
            logger.error(f"错误: [API请求] {str(e)}")
            raise


class FeishuSheets(FeishuBase):
    """飞书表格操作类

    功能分类:
    1. 表格基本操作 - Sheet Basic Operations
    2. 表格数据操作 - Sheet Data Operations
    3. 表格样式操作 - Sheet Style Operations
    4. 权限管理 - Permission Management
    5. 媒体操作 - Media Operations
    """

    def __init__(self):
        """初始化飞书表格操作类"""
        super().__init__()
        self.AUTH_URL = f'{self.BASE_URL}/auth/v3/tenant_access_token/internal/'
        self.SHEETS_URL = f'{self.BASE_URL}/sheets/v2/spreadsheets'
        self.SHEETS_V3_URL = f'{self.BASE_URL}/sheets/v3/spreadsheets'

    # ===================== 1. 表格基本操作 =====================

    def create_sheets(self, sheets_name: str = 'test') -> str:
        """创建飞书表格
        :param sheets_name: 表格名称
        :return: 创建的表格URL
        """
        body = {'title': sheets_name, 'type': 'sheet'}
        response = self._make_request('POST', self.SHEETS_V3_URL, json=body)
        sheet_url = response['data']['spreadsheet']['url']
        logger.info(f"操作成功: 创建飞书表格 {sheet_url}")
        return sheet_url

    def add_sheet(self, sheets_id: str, title: str) -> str:
        """添加新的sheet
        :param sheets_id: 表格ID
        :param title: sheet标题
        :return: 新创建的sheet ID
        """
        url = f'{self.SHEETS_URL}/{sheets_id}/sheets_batch_update'
        payload = {
            'requests': [
                {'addSheet': {'properties': {'title': title}}}
            ]
        }
        response = self._make_request('POST', url, json=payload)
        sheet_id = response['data']['replies'][0]['addSheet']['properties']['sheetId']
        logger.info('操作成功: 添加Sheet')
        return sheet_id

    def del_sheet(self, sheets_id: str, sheet_id: str):
        """删除sheet
        :param sheets_id: 表格ID
        :param sheet_id: 要删除的sheet ID
        """
        url = f'{self.SHEETS_URL}/{sheets_id}/sheets_batch_update'
        payload = {'requests': [{'deleteSheet': {'sheetId': sheet_id}}]}
        self._make_request('POST', url, json=payload)
        logger.info('操作成功: 删除Sheet')

    def read_sheets_metainfo(self, sheets_id: str) -> Dict:
        """读取表格元数据
        :param sheets_id: 表格ID
        :return: 表格元数据字典
        """
        url = f"{self.SHEETS_URL}/{sheets_id}/metainfo"
        response = self._make_request('GET', url, params={'extFields': 'protectedRange'})
        logger.info('操作成功: 读取表格元数据')
        return response['data']

    # ===================== 2. 表格数据操作 =====================
    def _chunk_dataframe(self, df: pd.DataFrame, chunk_size: int = 4000) -> List[pd.DataFrame]:
        """DataFrame分片处理
        :param df: 需要分片的DataFrame
        :param chunk_size: 每片的最大行数
        :return: 分片后的DataFrame列表
        """
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    def cover_sheet(self, sheets_id: str, sheet_id: str, df: pd.DataFrame):
        """覆盖写入表格数据
        :param sheets_id: 表格ID
        :param sheet_id: sheet ID
        :param df: 要写入的数据DataFrame
        """
        url = f"{self.SHEETS_URL}/{sheets_id}/values"
        df = df.reset_index(drop=True).fillna('')
        chunks = self._chunk_dataframe(df)

        for i, chunk in enumerate(chunks):
            is_first_chunk = i == 0
            if is_first_chunk:
                range_str = f'{sheet_id}!A1:Z{chunk.shape[0] + 1}'
                values = [chunk.columns.tolist()] + chunk.values.tolist()
            else:
                start_row = i * 4000 + 2
                end_row = start_row + chunk.shape[0]
                range_str = f'{sheet_id}!A{start_row}:Z{end_row}'
                values = chunk.values.tolist()

            data = {
                "valueRange": {
                    "range": range_str,
                    "values": values
                }
            }

            self._make_request('PUT', url, json=data)
            logger.info(f'操作成功: 导入第{i + 1}个数据块')

    def read_sheet(self, sheets_id: str, sheet_id: str) -> pd.DataFrame:
        """读取飞书表格数据
        :param sheets_id: 表格ID
        :param sheet_id: sheet ID
        :return: 包含表格数据的DataFrame
        """
        url = f"{self.SHEETS_URL}/{sheets_id}/values/{sheet_id}"
        response = self._make_request('GET', url)
        values = response['data']["valueRange"]["values"]
        if not values:
            logger.warning('警告: 未获取到表格数据')
            return pd.DataFrame()
        df = pd.DataFrame(data=values[1:], columns=values[0])
        logger.info('操作成功: 获取飞书表格数据')
        return df

    def append_sheet(self, sheets_id: str, sheet_id: str, df: pd.DataFrame):
        """追加数据到表格
        :param sheets_id: 表格ID
        :param sheet_id: sheet ID
        :param df: 要追加的数据DataFrame
        """
        url = f"{self.SHEETS_URL}/{sheets_id}/values"
        df = df.fillna('')
        chunks = self._chunk_dataframe(df)

        for i, chunk in enumerate(chunks):
            range_str = f'{sheet_id}!A:Z'
            values = chunk.values.tolist()

            data = {
                "valueRange": {
                    "range": range_str,
                    "values": values
                }
            }

            self._make_request('POST', url, params={'valueRangeOption': 'APPEND_ROW'}, json=data)
            logger.info(f'操作成功: 追加第{i + 1}个数据块')

    # ===================== 3. 表格样式操作 =====================

    def change_sheet_style(self, sheets_id: str, sheet_id: str, range_str: Optional[str] = None,
                           style: Optional[Dict] = None):
        """设置单元格样式
        :param sheets_id: 表格ID
        :param sheet_id: sheet ID
        :param range_str: 单元格范围，如 'A1:B2'
        :param style: 样式配置字典
        :return: None
        """
        url = f'{self.SHEETS_URL}/{sheets_id}/style'

        range_str = range_str or f"{sheet_id}!A1:T1"

        style = style or {
            "font": {"bold": True},
            "hAlign": 1
        }

        body = {
            "appendStyle": {
                "range": range_str,
                "style": style
            }
        }

        self._make_request('PUT', url, json=body)
        logger.info('操作成功: 设置单元格样式')

    def forzen_sheet(self, sheets_id: str, sheet_id: str, rowCount: int = 0, colCount: int = 0):
        """冻结窗口
        :param sheets_id: 表格ID
        :param sheet_id: sheet ID
        :param rowCount: 冻结的行数
        :param colCount: 冻结的列数
        :return: None
        """
        url = f'{self.SHEETS_URL}/{sheets_id}/sheets_batch_update'
        payload = {
            'requests': [{
                'updateSheet': {
                    'properties': {
                        'sheetId': sheet_id,
                        'frozenRowCount': rowCount,
                        'frozenColCount': colCount
                    }
                }
            }]
        }
        self._make_request('POST', url, json=payload)
        logger.info('操作成功: 冻结窗口')

    def update_sheet(self, sheets_id: str, params: Dict):
        """修改表格属性
        :param sheets_id: 表格ID
        :param params: 要更新的属性字典
        """
        url = f'{self.SHEETS_URL}/{sheets_id}/sheets_batch_update'
        payload = {'requests': [{'updateSheet': params}]}
        self._make_request('POST', url, json=payload)
        logger.info('操作成功: 更新工作表属性')

    # ===================== 4. 权限管理 =====================

    def get_user_id(self, user_id_type='open_id', in_type='emails', in_list=['lvjingxuan@chagee.com']) -> pd.DataFrame:
        """批量获取用户id
        :param user_id_type: 返回的用户ID类型
        :param in_type: 输入的ID类型
        :param in_list: 用户ID列表
        :return: 包含用户ID信息的DataFrame
        """
        url = f"{self.BASE_URL}/contact/v3/users/batch_get_id?user_id_type={user_id_type}"
        chunks = [in_list[i:i + 50] for i in range(0, len(in_list), 50)]
        df = None

        for chunk in chunks:
            payload = {in_type: chunk}
            response = self._make_request('POST', url, json=payload)
            df_tmp = pd.DataFrame(response['data']['user_list'])
            df = pd.concat([df, df_tmp], axis=0) if df is not None else df_tmp
            logger.info('操作成功: 获取用户ID信息')

        return df.reset_index(drop=True)

    def change_owner_power(self, sheets_id: str, type='sheet', email='lvjingxuan@chagee.com'):
        """转移文档所有者权限
        :param sheets_id: 表格ID
        :param type: 文档类型
        :param email: 新所有者邮箱
        """
        url = f'{self.BASE_URL}/drive/v1/permissions/{sheets_id}/members/transfer_owner'
        body = {
            'member_type': 'email',
            'member_id': email
        }
        self._make_request('POST', url, params={'type': type}, json=body)
        logger.info('操作成功: 转移文档所有者权限')

    def change_sheet_power(self, sheets_id: str, email: str = "lvjingxuan@chagee.com"):
        """修改表格权限为完全访问
        :param sheets_id: 表格ID
        :param email: 用户邮箱
        """
        url = f"{self.BASE_URL}/drive/v1/permissions/{sheets_id}/members"
        body = {
            "member_type": "email",
            "member_id": email,
            "perm": "full_access"
        }
        self._make_request('POST', url, params={'type': 'sheet', 'need_notification': True}, json=body)
        logger.info('操作成功: 设置完全访问权限')

    def change_sheet_power03(self, sheets_id: str, email: str = "lvjingxuan@chagee.com"):
        """修改表格编辑权限
        :param sheets_id: 表格ID
        :param email: 用户邮箱
        """
        url = f"{self.BASE_URL}/drive/v1/permissions/{sheets_id}/members"
        body = {
            "member_type": "email",
            "member_id": email,
            "perm": "edit"
        }
        self._make_request('POST', url, params={'type': 'sheet', 'need_notification': True}, json=body)
        logger.info('操作成功: 修改表格编辑权限')

    # ===================== 6. 媒体操作 =====================

    def upload_image(self, image_path: str) -> str:
        """上传图片到飞书
        :param image_path: 图片文件路径
        :return: 上传后的文件token
        """
        url = f"{self.BASE_URL}/drive/v1/medias/upload_all"

        with open(image_path, 'rb') as f:
            file_content = f.read()

        form = {
            'file_name': image_path.split('/')[-1],
            'parent_type': 'explorer',
            'parent_node': 'root',
            'size': str(len(file_content)),
            'file': (image_path.split('/')[-1], file_content, 'image/png')
        }

        multipart = MultipartEncoder(fields=form)
        headers = self.header.copy()
        headers['Content-Type'] = multipart.content_type

        response = requests.post(url, headers=headers, data=multipart).json()

        if response['code'] != 0:
            error_msg = f"code={response['code']}, msg={response.get('msg', 'Unknown error')}"
            logger.error(f"错误: [上传图片] {error_msg}")
            raise Exception(error_msg)

        file_token = response['data']['file_token']
        logger.info('操作成功: 上传图片')
        return file_token

if __name__ == '__main__':
    fs = FeishuSheets()
    df = fs.read_sheet('Eeg0sMZxFhncgxtVGeTckp6qnqb', '550ed9')
    print(df)


