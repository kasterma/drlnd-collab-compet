"""
Module defining global configuration for the collab-compet package
"""

from logging.config import dictConfig
import yaml
import torch

# noinspection PyUnresolvedReferences
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
    dictConfig(log_conf)

with open("config.yaml") as conf_file:
    config = yaml.load(conf_file)

config['database_file'] = "data/rundb.sqlite"

print("laod config")
