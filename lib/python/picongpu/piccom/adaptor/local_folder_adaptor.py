"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from enum import Enum
from typing import Any

import numpy as np

from picongpu.piccom.schema.info import RuntimeInfo
from picongpu.piccom.schema.metadata_file import MetadataFile


def _contains_single_value(lhs, rhs):
    if isinstance(rhs, slice):
        return lhs >= (rhs.start or -np.inf) and lhs <= (rhs.stop or np.inf)
    return lhs == rhs


def _contains(content, parameters):
    return all(key in content and _contains_single_value(content[key], value) for key, value in parameters.items())


ACTION_NAME = {"parameters": "generate_input_files", "runtime_info": "run"}


def _extract(content: MetadataFile | dict, what) -> dict[str, Any]:
    if isinstance(content, dict):
        return _extract_parameters(MetadataFile.model_validate(content))
    return next(filter(lambda x: x.action_name == ACTION_NAME[what], content.log.values())).content


def _extract_parameters(content: MetadataFile | dict) -> dict[str, Any]:
    return _extract(content, "parameters")


def _extract_runtime_info(content: MetadataFile | dict) -> RuntimeInfo:
    return RuntimeInfo(**_extract(content, "runtime_info"))


def _filter_by_parameters(objs, parameters):
    if parameters is None:
        return objs
    return filter(lambda obj: _contains(_extract_parameters(obj), parameters), objs)


class LoggedStatus(Enum):
    success = "success"
    failure = "failure"
    none = None


class Status(Enum):
    success = "success"
    failure = "failure"
    started = "started"
    running = "running"
    ended = "ended"


def _get_status(i, log) -> LoggedStatus:
    statuses = set(map(lambda obj: obj["action_name"], filter(lambda obj: obj["update_of"] == i, log.values())))
    if "success" in statuses:
        return LoggedStatus.success
    if "failure" in statuses:
        return LoggedStatus.failure
    if len(statuses) == 0:
        return LoggedStatus.none
    raise ValueError(f"Couldn't find out status of {i=} in {log=}.")


def _match_status(s1: LoggedStatus, s2: Status):
    if s2 == Status.success:
        return s1 == LoggedStatus.success
    if s2 == Status.failure:
        return s1 == LoggedStatus.failure
    if s2 == Status.started:
        # We have got an s1, so it has sure started at some point.
        return True
    if s2 == Status.running:
        return s1 == LoggedStatus.none
    if s2 == Status.ended:
        return s1 != LoggedStatus.none
    raise ValueError(f"Matching {s1=} and {s2=} ended up in an unreachable code path.")


def _has_status(log, status):
    actions = [
        (map(lambda action: action[0], filter(lambda action: action[1]["action_name"] == name, log.items())), s)
        for name, s in status.items()
    ]
    return all(any(filter(lambda i: _match_status(_get_status(i, log), Status(s)), ids)) for ids, s in actions)


def _filter_by_status(objs, status):
    if status is None:
        return objs
    return filter(lambda obj: _has_status(obj["log"], status), objs)


def _extract_id(obj):
    return obj["_id"]


class LocalFolderAdaptor:
    def __init__(self, database):
        self.database = database

    def get_ids(self, parameters: dict[str, Any] | None = None, status: dict[str, str] | None = None):
        return list(
            map(_extract_id, _filter_by_parameters(_filter_by_status(self.database.find(), status), parameters))
        )
