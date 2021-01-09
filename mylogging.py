import logging


class log():
    def __init__(self,log_name,file_path):
        # 获取logger对象,取名mylog
        self.logger = logging.getLogger(log_name)
        # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
        self.logger.setLevel(level=logging.DEBUG)

        # 获取文件日志句柄并设置日志级别，第二层过滤
        self.handler = logging.FileHandler(file_path)
        self.handler.setLevel(logging.INFO)  

        # 生成并设置文件日志格式，其中name为上面设置的mylog
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)

        # 获取流句柄并设置日志级别，第二层过滤
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.WARNING)

        # 为logger对象添加句柄
        self.logger.addHandler(self.handler)
        self.logger.addHandler(self.console)
        
    def log_info(self,msg):
        # 记录日志
        self.logger.info(msg)
        
        
    def log_debug(self,msg):
        self.logger.debug(msg)
    
    def log_warning(self,msg):
        self.logger.warning(msg)
