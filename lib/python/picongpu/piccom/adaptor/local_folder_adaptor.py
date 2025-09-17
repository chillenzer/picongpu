"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from enum import Enum
from typing import Any, Iterable
from random import sample
from itertools import batched

import numpy as np

from picongpu.piccom.schema.info import RuntimeInfo
from picongpu.piccom.schema.metadata_file import MetadataFile


class Status(Enum):
    success = "success"
    failure = "failure"
    started = "started"
    running = "running"
    ended = "ended"


def _contains_single_value(lhs, rhs):
    if isinstance(rhs, slice):
        return lhs >= (rhs.start or -np.inf) and lhs <= (rhs.stop or np.inf)
    return lhs == rhs


def _contains(content, parameters):
    return all(key in content and _contains_single_value(content[key], value) for key, value in parameters.items())


_ACTION_NAME = {"parameters": "generate_input_files", "runtime_info": "run"}


def _extract(content: MetadataFile | dict, what) -> dict[str, Any]:
    if isinstance(content, dict):
        return _extract_parameters(MetadataFile.model_validate(content))
    return next(filter(lambda x: x.action_name == _ACTION_NAME[what], content.log.values())).content


def _extract_parameters(content: MetadataFile | dict) -> dict[str, Any]:
    return _extract(content, "parameters")


def _extract_runtime_info(content: MetadataFile | dict) -> RuntimeInfo:
    return RuntimeInfo(**_extract(content, "runtime_info"))


def _filter_by_parameters(objs, parameters):
    if parameters is None:
        return objs
    return filter(lambda obj: _contains(_extract_parameters(obj), parameters), objs)


class _LoggedStatus(Enum):
    success = "success"
    failure = "failure"
    none = None


def _get_status(i, log) -> _LoggedStatus:
    statuses = set(map(lambda obj: obj["action_name"], filter(lambda obj: obj["update_of"] == i, log.values())))
    if "success" in statuses:
        return _LoggedStatus.success
    if "failure" in statuses:
        return _LoggedStatus.failure
    return _LoggedStatus.none


def _match_status(s1: _LoggedStatus, s2: Status):
    if s2 == Status.success:
        return s1 == _LoggedStatus.success
    if s2 == Status.failure:
        return s1 == _LoggedStatus.failure
    if s2 == Status.started:
        # We have got an s1, so it has sure started at some point.
        return True
    if s2 == Status.running:
        return s1 == _LoggedStatus.none
    if s2 == Status.ended:
        return s1 != _LoggedStatus.none
    raise ValueError(f"Matching {s1=} and {s2=} ended up in an unreachable code path.")


def _has_status(log, status):
    actions = [
        # This needs to be a list because we need to eagerly bind the `name` variable in the loop.
        # If the lambdas are lazily evaluated, they'll all bind to the last value of `name`.
        (list(map(lambda action: action[0], filter(lambda action: action[1]["action_name"] == name, log.items()))), s)
        for name, s in status.items()
    ]
    return all(any(filter(lambda i: _match_status(_get_status(i, log), Status(s)), ids)) for ids, s in actions)


def _filter_by_status(objs, status):
    if status is None:
        return objs
    return filter(lambda obj: _has_status(obj["log"], status), objs)


def _extract_id(obj):
    return obj["_id"]


class Ordering(Enum):
    arbitrary = "arbitrary"
    random = "random"


def _apply_ordering(objs, ordering: Ordering):
    if ordering == Ordering.arbitrary:
        return objs
    if ordering == Ordering.random:
        # for random access we actually have to hold a copy in memory
        objs = list(objs)
        return iter(sample(objs, k=len(objs)))
    raise ValueError(f"Applying the {ordering=} to {objs=} reached an unreachable branch.")


def _apply_batch_size(objs, batch_size: int | None):
    return objs if batch_size is None else batched(objs, batch_size)


def _make_list(iterable, recursion_depth: int = 0):
    if recursion_depth < 0:
        raise ValueError("You're running into an infinite recursion because {recursion_depth=} < 0.")
    if recursion_depth == 0:
        return list(iterable)
    return [_make_list(i, recursion_depth - 1) for i in iterable]


def _extract_full_metadata(obj):
    return obj


class Retrievable(Enum):
    full_metadata = "full_metadata"
    ids = "ids"
    parameters = "parameters"


_RETRIEVABLE_TO_FUNCTION = {
    Retrievable.full_metadata: _extract_full_metadata,
    Retrievable.ids: _extract_id,
    Retrievable.parameters: _extract_parameters,
}


class AllIds:
    def __contains__(self, _):
        return True


def _filter_by_ids(objs, ids):
    return filter(lambda obj: _extract_id(obj) in ids, objs)


def _get_full_metadata(
    database,
    ids: Iterable[str] | None = None,
    parameters: dict[str, Any] | None = None,
    status: dict[str, str] | None = None,
    ordering: Ordering = Ordering.arbitrary,
):
    return _apply_ordering(
        _filter_by_parameters(_filter_by_status(_filter_by_ids(database.find(), ids or AllIds()), status), parameters),
        Ordering(ordering),
    )


def _get_batched_information(database, information, batch_size: int | None = None, **kwargs):
    return _apply_batch_size(
        map(information, _get_full_metadata(database, **kwargs)),
        batch_size,
    )


class LocalFolderAdaptor:
    def __init__(self, database):
        self.database = database

    def get(
        self,
        retrievable: Retrievable | str,
        return_as_iterators: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ):
        retrievable = Retrievable(retrievable)
        if not return_as_iterators:
            return _make_list(
                self.get(retrievable, return_as_iterators=True, batch_size=batch_size, **kwargs),
                0 if batch_size is None else 1,
            )
        return _get_batched_information(
            self.database, _RETRIEVABLE_TO_FUNCTION[retrievable], batch_size=batch_size, **kwargs
        )
