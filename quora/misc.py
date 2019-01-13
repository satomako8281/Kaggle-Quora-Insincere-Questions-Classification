import socket
import json
import logging
import requests
import time
from contextlib import contextmanager

import numpy as np


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
    print_(f'[{name}] start')
    yield
    print_(f'[{name}] done in {time.time() - t0:.0f} s')


def timestamp():
    return time.strftime('%y%m%d_%H%M%S')
