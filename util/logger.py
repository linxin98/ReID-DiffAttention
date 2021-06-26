import logging


def get_logger(level='INFO'):
    level_relations = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    logger = logging.getLogger()

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(module)s %(levelname)s: %(message)s')

    logger.setLevel(level_relations[level])
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
