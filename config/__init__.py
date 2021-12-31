import os
import yaml


def init_config(cfg):
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../config')
    with open(os.path.join(basedir, cfg), 'r') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)
    return conf


conf = init_config("homo.yml")

