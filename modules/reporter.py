import os
from enum import Flag, auto
import logging
from typing import List, Union
from dataclasses import is_dataclass, asdict
from logging.handlers import RotatingFileHandler
import json

try:
    import neptune
except ModuleNotFoundError:
    pass
try:
    import comet_ml as comet
except ModuleNotFoundError:
    pass

_logger = logging.getLogger(__name__)
_configs = {}
_all_parameters = {}


class Backend(Flag):
    Void = 0
    Logging = auto()
    Comet = auto()
    Neptune = auto()


def init(project_name: str, backend: Backend = Backend.Void, debug=False):
    _configs["backend"] = backend
    _configs["debug"] = debug
    _configs["project_name"] = project_name
    project_name = project_name.replace(" ", "_")
    if backend & Backend.Neptune:
        neptune_configs = {
            "project_qualified_name": f"tihbe/{project_name}",
            "api_token": os.environ["NEPTUNE_API_TOKEN"],
            "backend": neptune.OfflineBackend() if debug else None,
        }
        neptune.init(**neptune_configs)
        experiment = neptune.create_experiment(upload_stdout=False, upload_stderr=True)
        _configs["neptune_experiment"] = experiment

    if backend & Backend.Comet:
        experiment = comet.Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name=project_name,
            workspace="tihbe",
            disabled=debug,
        )
        _configs["comet_experiment"] = experiment

    if backend & Backend.Logging:
        logfile = os.path.join(
            os.getcwd(), f"{project_name}{'_DEBUG' if debug else ''}.log"
        )
        if os.path.isfile(logfile):  # Roll logs if file exist ;; up to 20 files
            temp_handler = logging.handlers.RotatingFileHandler(logfile, backupCount=20)
            temp_handler.doRollover()
            temp_handler.close()

        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=logfile,
        )


def _assert_backend():
    if not "backend" in _configs:
        raise Exception("Reporter was not initialized")


def log_parameters(parameters: Union[dict, object]):
    _assert_backend()

    params = parameters
    if is_dataclass(parameters):
        params = asdict(params)

    if _configs["backend"] & Backend.Comet:
        _configs["comet_experiment"].log_parameters(params)

    for k, v in params.items():
        _all_parameters[k] = str(v)
        if _configs["backend"] & Backend.Neptune:
            _configs["neptune_experiment"].set_property(k, v)
        if _configs["backend"] & Backend.Logging:
            _logger.info(f"PARAMETER '{k}'='{v}'")

    return parameters


def log_parameter(name, value):
    log_parameters({name: value})
    return value


def log_tags(tags: List[str]):
    _assert_backend()

    if _configs["backend"] & Backend.Comet:
        _configs["comet_experiment"].add_tags(tags)

    if _configs["backend"] & Backend.Neptune:
        _configs["neptune_experiment"].append_tags(*tags)

    for tag in tags:
        if _configs["backend"] & Backend.Logging:
            _logger.info(f"TAGGED '{tag}'")

    return tags


def log_tag(tag: str):
    log_tags([tag])
    return tag


def log_metrics(metrics: dict):
    _assert_backend()
    if _configs["backend"] & Backend.Comet:
        _configs["comet_experiment"].log_metrics(metrics)

    for k, v in metrics.items():
        if _configs["backend"] & Backend.Neptune:
            _configs["neptune_experiment"].log_metric(k, v)

        if _configs["backend"] & Backend.Logging:
            _logger.info(f"METRIC '{k}'='{v}'")


def log_metric(name: str, value):
    log_metrics({name: value})
    return value


def dump_results(**results):
    full_path = os.path.join(os.getcwd(), f"{_configs['project_name']}_results.json")
    if os.path.isfile(full_path):  # Roll experiments if file exist ;; up to 20 files
        temp_handler = logging.handlers.RotatingFileHandler(full_path, backupCount=20)
        temp_handler.doRollover()
        temp_handler.close()

    file_handler = open(full_path, "a")
    json.dump({**_all_parameters, **results}, file_handler)
