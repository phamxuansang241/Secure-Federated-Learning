import functools
import time


def timer(func):
    """
    Print the runtime of the decorated function
    """
    @functools.wraps(func)
    def wrapper_time(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("\t\t Total time taken to train: {:.4f}s".format(run_time))

        return value

    return wrapper_time


def print_decorator(epoch):

    def inner(func):
        def wrapper(*args, **kwargs):
            print('[TRAINING] Global epoch {} starts ...'.format(epoch))
            result = func(*args, **kwargs)
            print('\t Loss (client mean): {}'.format(result))

            return result
        return wrapper

    return inner
