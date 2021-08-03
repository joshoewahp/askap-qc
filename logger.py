import colorlog
import logging

def setupLogger(verbose: bool, extra_verbose: bool = False, filename: str = None):

    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not extra_verbose:
        # Turn off some bothersome verbose logging modules
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

    if filename:
        formatter = logging.Formatter(
            '%(levelname)-8s %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    colorformatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(asctime)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white', })

    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(colorformatter)

    root_logger.addHandler(stream_handler)
