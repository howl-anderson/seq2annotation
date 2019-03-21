class TaskStatus(object):
    def __init__(self, config):
        self.DONE = 10
        self.START = 1

    def send_status(self, status):
        print('{}:{}'.format(self.__class__, status))
