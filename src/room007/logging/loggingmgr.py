# vim: set fileencoding=utf-8 :

from logging.config import fileConfig
from os.path import dirname, join

__set_up = False


def set_up(name='default'):
    global __set_up
    if not __set_up:
        fileConfig(join(dirname(__file__), name + '.cfg'),
                   disable_existing_loggers=False)
        __set_up = True
