import functools
import time
from threading import Thread
import os

from flask import Flask


def run_function_in_background(target):
    t = Thread(target=target)
    t.setDaemon(True)
    t.start()


def http_transponder(port):
    app = Flask(__name__)

    @app.route("/")
    def main():
        return 'I am the health check transponder'

    @app.route("/ping")
    def ping():
        return 'pong'

    @app.route("/are_you_ok")
    def are_you_ok():
        return "I'm OK"

    app.run(host='0.0.0.0', port=port)


# try get port config from environment
http_port = os.getenv('HEALTH_CHECK_TRANSPONDER_PORT', 9998)


run_health_check_transponder_in_background = functools.partial(
    run_function_in_background,
    functools.partial(http_transponder, port=http_port)
)


if __name__ == "__main__":
    run_health_check_transponder_in_background()

    for _ in range(100):
        time.sleep(1)
        print('.', end='', flush=True)
