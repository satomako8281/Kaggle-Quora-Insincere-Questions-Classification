import socket
import json
import logging
import requests
import time
from contextlib import contextmanager

import numpy as np
from quora.config import logger


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def write_spreadsheet(*args):
    """
    # scores = np.array([0.8, 0.7, 0.75, 0.8]) とする
    # write_spreadsheet('baseline', 0.8, 0.7, 0.75, 0.8) は以下のように書ける
    write_spreadsheet('baseline', *scores)
    """
    endpoint = 'https://script.google.com/macros/s/AKfycbz-_peU1j6S6kkJPDqsM1XJ1AwLFdLzIx7gUVxkjsnxgXrY8ZM/exec'
    requests.post(endpoint, json.dumps(args))


def send_line_notification(message):
    line_token = 'LLT1Huejwe1C5IPxcLoIi2mZWelUc0QpGEJjpqrXucH'  # 終わったら無効化する
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}\n{}".format(socket.gethostname(), message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

@contextmanager
def timer(name, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_('[{}] start'.format(name))
    yield
    print_('[{}] done in {:.0f} s'.format(name, time.time() - t0))


def timestamp():
    return time.strftime('%y%m%d_%H%M%S')


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        logger.info('Starting {}'.format(self.message))
        self.start_clock = time.clock()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_clock = time.clock()
        self.end_time = time.time()
        self.interval_clock = self.end_clock - self.start_clock
        self.interval_time = self.end_time - self.start_time
        logger.info('Finished {}. Took {:.2f} seconds, CPU time {:.2f}, effectiveness {:.2f}'.format(
            self.message, self.interval_time, self.interval_clock, self.interval_clock / self.interval_time))
