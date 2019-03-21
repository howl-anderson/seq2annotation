class PerformanceMetrics(object):
    def __init__(self, config):
        pass

    def set_metrics(self, name, metrics):
        print('{}: {} => {}'.format(self.__class__, name, metrics))
