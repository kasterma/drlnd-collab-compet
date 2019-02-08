"""
Module defining global configuration for the collab-compet package
"""

from logging.config import dictConfig
import yaml
import torch
import logging

# noinspection PyUnresolvedReferences
from collabcompet.orm import session, Run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
    dictConfig(log_conf)

log = logging.getLogger("config")

with open("config.yaml") as conf_file:
    config = yaml.load(conf_file)
    log.info(f"Running with config: {config}")


def load_config_from_db(run_id: int):
    global config
    runs = session.query(Run).filter(Run.id == run_id).all()
    assert len(runs) == 1
    config = yaml.load(runs[0].config)
    log.info(f"Loaded config: {config}")
