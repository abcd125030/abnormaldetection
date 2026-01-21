import pandas as pd
from loguru import logger
from sqlalchemy import create_engine


class DbSqlalchemy:
    def __init__(self):
        self.engine = None

    # 换成sqlalchemy的create_engine
    def dbSetup(self, host, port, username, password, database):
        try:
            # 创建数据库连接引擎
            self.engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}?connect_timeout=30')
            logger.bind(decorHaier=True).info('数据库连接成功')
        except Exception as e:
            logger.bind(decorHaier=True).info('数据库连接失败！错误信息是%s' % e)
            raise Exception('数据库连接失败！错误信息是%s' % e)