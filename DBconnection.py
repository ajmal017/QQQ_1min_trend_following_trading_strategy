from sqlalchemy import create_engine

class DBconnection():
    def __init__(self, _type, _driver, _user, _password, _host, _port, _dbname, _poolsize = 50):
        self.type = _type
        self.driver = _driver
        self.user = _user
        self.password = _password
        self.host = _host
        self.port = _port
        self.dbname = _dbname
        SQLALCHEMY_DATABASE_URL = '%s+%s://%s:%s@%s:%s/%s' % (_type, _driver, _user, _password, _host, _port, _dbname)
        self.ENGINE = create_engine(SQLALCHEMY_DATABASE_URL, pool_size = _poolsize, max_overflow = 0)


    def DBconnect(self):
        conn = self.ENGINE.connect()
        return conn