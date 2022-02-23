import logging


def get_logger(level='INFO'):
    level_relations = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    # Set logger level.
    logger = logging.getLogger()
    logger.setLevel(level_relations[level])
    # Format logger output.
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
