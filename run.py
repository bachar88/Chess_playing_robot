import eventlet
eventlet.monkey_patch()

from chess_server import start_server
from Main import main as cv_main

if __name__ == '__main__':
    start_server()
    cv_main()