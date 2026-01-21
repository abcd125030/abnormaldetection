import records
from loguru import logger


class DbAdapter:
    def __init__(self):
        self.db_pool = None

    def dbSetup(self, host, port, user, password, database):
        try:
            self.db_pool = records.Database(
                db_url=f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8',
                pool_recycle=True,  # 多久之后对连接池中连接进行一次回收
                max_overflow=200,  # 超过连接池大小之后，允许最大扩展连接数
                pool_size=128,  # 连接池的大小
                pool_timeout=10,  # 连接池如果没有连接了，最长的等待时间
            )
            '''
            db_connect = aiomysql.create_pool(host=DBConfig.DB_HOST,
                                               port=DBConfig.DB_PORT,
                                               user=DBConfig.DB_ACCOUNT,
                                               password=DBConfig.DB_PWD,
                                               db=DBConfig.DB_NAME,
                                               loop=loop)
            '''
        except Exception as e:
            logger.bind(decorHaier=True).info('数据库连接失败！错误信息是%s' % e)
            raise Exception('数据库连接失败！错误信息是%s' % e)

    def dbQuery(self, sql):
        try:
            structure_results = self.db_pool.query(sql).all(as_dict=True)
            return structure_results
        except Exception as e:
            logger.bind(decorHaier=True).info('sql语句错误，%s\n，错误的sql是【%s】' % (e, sql))
            raise Exception('sql语句错误，%s\n，错误的sql是【%s】' % (e, sql))
