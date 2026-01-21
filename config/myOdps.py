from loguru import logger
from odps import ODPS
from config import settings
from odps import options


class PYODPS:
    def __init__(self, point='shanghai'):
        config = settings['dataworks'][point]
        self.odps_config = {
            'access_id': config['access_id'],
            'secret_access_key': config['secret_access_key'],
            'endpoint': config['endpoint'],
            'project': 'chagee_data_warehouse',
        }

    def get_odps(self):
        """
        获取odps
        """
        try:
            return ODPS(**self.odps_config)
        except Exception as e:
            logger.bind(decorChaGee=True).error(f"Failed to connect to ODPS:{e}")

    def execute_sql(self, sql):
        """
        执行sql
        :param sql:
        """
        try:
            self.get_odps().execute_sql(sql)
            # logger.bind(decorChaGee=True).info('successfully.')
        except Exception as e:
            logger.bind(decorChaGee=True).error(f"error: {e}")

    def get_table(self, table_name):
        """
        获取表信息
        :param table_name
        """
        try:
            c = self.get_odps().get_table(table_name)
            logger.bind(decorChaGee=True).info(c)
        except Exception as e:
            logger.bind(decorChaGee=True).error(f'error:{e}')

    def select_table(self, sql):
        """
        tunnel读表
        :param sql
        """
        options.verbose = True
        try:
            with self.get_odps().execute_sql(sql).open_reader(tunnel=True, limit=False) as reader:
                df = reader.to_pandas()
            # logger.bind(decorChaGee=True).info('successfully.')
            return df
        except Exception as e:
            logger.bind(decorChaGee=True).error(f'error:{e}')
