# vim: set fileencoding=utf-8 :

import time


def time_function(logger, conservative=False):
    def wrap(fun):
        def wrapped(*args, **kwargs):
            logger.info('called %s', fun.__name__)
            start_time = time.time()
            returns = fun(*args, **kwargs)
            end_time = time.time()
            time_needed = end_time - start_time
            logger.info('running %s took %#.3f s', fun.__name__, time_needed)
            return returns if conservative else (returns, time_needed)
        return wrapped
    return wrap
