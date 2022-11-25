# todo: @gin.configurable
class Config:
    def try_load_configfile(self):
        # todo: try load from env var PRL_CONFIG
        pass

    def __init__(self):
        self.try_load_configfile()

