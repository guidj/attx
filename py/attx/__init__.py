import os
import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
        'tensorflow': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
})

logger = logging.getLogger('attx')

PY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PY_DIR, '../../')
