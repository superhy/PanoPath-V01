'''
Created on 24 Apr 2024

@author: yang hu
'''
import sys


class Logger(object):

    def __init__(self, filename='running_log-default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    pass