import logging


def save_training_log(log_freq, step, batch_size, prefix: str = '', **kwargs):
    if log_freq is None:
        return
    elif log_freq == 0 and step == batch_size - 1:
        pass
    elif log_freq > 0 and step % log_freq != 0:
        pass
