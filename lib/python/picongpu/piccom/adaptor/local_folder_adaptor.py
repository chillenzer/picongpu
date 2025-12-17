"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from ast import literal_eval
from enum import Enum
from functools import reduce
from random import choices, sample
from typing import Any, Callable, Iterable

import numpy as np

from picongpu.piccom.schema.info import RuntimeInfo
from picongpu.piccom.schema.metadata_file import MetadataFile


# One would think that
#     from itertools import batched
# could be a better idea but it returns the batches as tuples
# which might not be what we want concerning the laziness
# and memory consumption of the algorithm.
class batched:
    def __init__(self, iterable, batch_size):
        self.batch_size = batch_size
        self.iterable = iter(iterable)
        self.exhausted = False
        self.previous_gen_exhausted = True

    def __next__(self):
        if not self.previous_gen_exhausted:
            raise ValueError(
                "The protocol for our purely iterator-based `batched` algorithm requires that you first exhaust the previous batch."
            )
        if self.exhausted:
            raise StopIteration()

        def gen(instance):
            for _ in range(instance.batch_size):
                try:
                    yield next(instance.iterable)
                except StopIteration:
                    instance.exhausted = True
                    break
            instance.previous_gen_exhausted = True

        self.previous_gen_exhausted = False
        return gen(self)

    def __iter__(self):
        return self


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


def _extract(content: MetadataFile | dict, what) -> dict[str, Any]:
    if isinstance(content, dict):
        return _extract(MetadataFile.model_validate(content), what)
    return reduce(lambda acc, rhs: acc | rhs.content, filter(lambda x: x.action_name == what, content.log.values()), {})


def _simplify_species(content):
    try:
        content["species"] = {species["name"]: species for species in content["species_initmanager"]["species"]}
    except Exception as err:
        logging.debug(str(err))
    return content


def _extract_parameters(content: MetadataFile | dict) -> dict[str, Any]:
    additional_parameters = _extract(content, "additional_parameters")
    return _simplify_species(
        (additional_parameters or {})
        | _extract(content, "generate_input_files")["simulation"]
        | ({"additional_parameters": additional_parameters} if additional_parameters else {})
    )


def _extract_runtime_info(content: MetadataFile | dict) -> RuntimeInfo:
    return RuntimeInfo(**_extract(content, "run")["info"])


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
    random_with_replacement = "random_with_replacement"
    random_without_replacement = "random_without_replacement"


def _apply_ordering(objs, ordering: Ordering | Callable[[dict[str, Any]], float]):
    if not isinstance(ordering, Ordering):
        try:
            return _apply_ordering(objs, Ordering(ordering))
        except ValueError:
            return iter(sorted(objs, key=lambda obj: ordering(_extract_parameters(obj))))

    if ordering == Ordering.arbitrary:
        return objs
    # for random access we actually have to hold a copy in memory
    # for printing the error message afterwards it's also more readable
    objs = list(objs)
    if ordering == Ordering.random_without_replacement:
        return iter(sample(objs, k=len(objs)))
    if ordering == Ordering.random_with_replacement:
        return iter(choices(objs, k=len(objs)))
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


class ExtractionFailure:
    pass


class NotFound(ExtractionFailure):
    pass


def _matches_wildcard(obj, wildcard):
    if wildcard == "":
        return True
    split_wc = wildcard.split("=", maxsplit=1)
    if len(split_wc) == 2:
        value = _extract_from(obj, split_wc[0])
        try:
            return value == literal_eval(split_wc[1])
        except ValueError:
            return False
    return not isinstance(_extract_from(obj, split_wc[0]), NotFound)


def _indefinite_wildcard_extract_from(obj, name):
    if isinstance(obj, dict):
        if _matches_wildcard(obj, name):
            return [_extract_from(obj, name)]
        obj = list(obj.values())

    if isinstance(obj, list):
        try:
            return sum(
                (
                    result
                    for o in obj
                    if not isinstance((result := _indefinite_wildcard_extract_from(o, name)), NotFound)
                ),
                [],
            )
        except TypeError:
            return NotFound()
    return NotFound()


def _wildcard_extract_from(obj, name):
    try:
        wc, remainder = name[1:].split("}")
    except ValueError as error:
        raise ValueError(f"{name=} starts with '{{' but does not contain matching '}}'.") from error

    extractor = (lambda x, _: x) if remainder == "" else _extract_from
    remainder = remainder[1:]

    if wc == "...":
        return _indefinite_wildcard_extract_from(obj, remainder)

    return [extractor(o, remainder) for o in obj if _matches_wildcard(o, wc)]


def _extract_from(obj, name):
    if name.startswith("{"):
        if "}" in name:
            return _wildcard_extract_from(obj, name)
        else:
            raise ValueError(f"Found '{{' without corresponding closing bracket '}}' in {name=}.")
    split_name = name.split("/", maxsplit=1)
    if isinstance(obj, list):
        split_name[0] = int(split_name[0])
    try:
        if len(split_name) == 2:
            return _extract_from(obj[split_name[0]], split_name[1])
        return obj[split_name[0]]
    except KeyError:
        return NotFound()


class Parameter:
    def __init__(self, name):
        self.name = name

    def extract_from(self, obj):
        return _extract_from(obj, self.name)


class Result:
    def __init__(self, name):
        self.name = name

    def extract_from(self, obj: RuntimeInfo):
        return _extract_from(obj.expected_results, self.name)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _retrievable_to_function(retrievable):
    if isinstance(retrievable, str):
        try:
            return _retrievable_to_function(Retrievable(retrievable))
        except ValueError:
            return _retrievable_to_function(Parameter(retrievable))
    if retrievable == Retrievable.full_metadata:
        return _extract_full_metadata
    if retrievable == Retrievable.ids:
        return _extract_id
    if retrievable == Retrievable.parameters:
        return _extract_parameters
    if isinstance(retrievable, Parameter):
        return lambda obj: retrievable.extract_from(_extract_parameters(obj))
    if isinstance(retrievable, Result):
        return lambda obj: retrievable.extract_from(_extract_runtime_info(obj))
    if is_iterable(retrievable):
        if hasattr(retrievable, "items"):
            return lambda obj: {key: _retrievable_to_function(r)(obj) for key, r in retrievable.items()}
        else:
            return lambda obj: [_retrievable_to_function(x)(obj) for x in retrievable]
    raise ValueError(f"Normalising {retrievable=} to a function reached an unreachable branch.")


class AllIds:
    def __contains__(self, _):
        return True


def _filter_by_ids(objs, ids):
    return filter(lambda obj: _extract_id(obj) in ids, objs)


class ExtractionFailureException(Exception):
    pass


def _raise_exception(*args):
    raise ExtractionFailureException(*args) from args[0]


class ToBeRemoved:
    pass


def _remove_if_any(*args):
    return ToBeRemoved()


class HandleExtractionFailures(Enum):
    raise_exception = _raise_exception
    remove_if_any = _remove_if_any


def _contains_extraction_failure(obj):
    if isinstance(obj, ExtractionFailure):
        return obj, "", obj
    if isinstance(obj, dict):
        try:
            key, (failure, path, _) = next(
                filter(lambda x: x[1], ((key, _contains_extraction_failure(value)) for key, value in obj.items()))
            )
            return failure, f"{key}/{path}", obj
        except StopIteration:
            return False
    if isinstance(obj, list):
        return _contains_extraction_failure({key: val for key, val in enumerate(obj)})
    return False


def _handle_extraction_failures(objs, handler):
    if isinstance(handler, HandleExtractionFailures):
        handler = handler.value
    return filter(
        lambda o: not isinstance(o, ToBeRemoved),
        map(lambda o: handler(*not_found) if (not_found := _contains_extraction_failure(o)) else o, objs),
    )


def _get_full_metadata(
    database,
    ids: Iterable[str] | None = None,
    parameters: dict[str, Any] | None = None,
    status: dict[str, str] | None = None,
    ordering: Ordering | Callable[[dict[str, Any]], float] = Ordering.arbitrary,
):
    return _apply_ordering(
        _filter_by_parameters(_filter_by_status(_filter_by_ids(database.find(), ids or AllIds()), status), parameters),
        ordering,
    )


def _get_batched_information(
    database,
    information,
    batch_size: int | None = None,
    handle_extraction_failures: HandleExtractionFailures
    | Callable[[ExtractionFailure, str, dict[str, Any]], Any] = HandleExtractionFailures.remove_if_any,
    **kwargs,
):
    return _apply_batch_size(
        _handle_extraction_failures(
            map(information, _get_full_metadata(database, **kwargs)),
            handle_extraction_failures,
        ),
        batch_size,
    )


def InstanceOrIterableOf(t):
    return t | Iterable[t]


class LocalFolderAdaptor:
    def __init__(self, database):
        self.database = database

    def get(
        self,
        retrievable: InstanceOrIterableOf(Retrievable | Parameter | str),
        *,
        batch_size: int | None = None,
        return_as_iterators: bool = False,
        **kwargs,
    ):
        """
        Extract some information from the database.

        This is a convenience interface to search our metadata database
        and extract information abiding by given constraints.

        What does the output look like?
        -------------------------------
        The returned information is always an iterable containing every match.
        By default, it is a list but you can use `return_as_iterators=True` to make it an iterator.
        In some scenarios, this avoids holding the data in memory which might be faster
        at the expense of it getting exhausted upon use.

        The returned information is controlled via the `retrievable`.
        Basic atoms of information are either
            - a `Retrievable`: some high-level information (like "parameters" or "ids") or
            - a `Parameter`: some specific parameter like "grid/cell_size_x_si"

        The code tries to convert to those from a given string,
        so you normally don't need to use the types explicitly.

        The structure of the `retrievable` determines how the information from each dataset is represented.
        You can provide iterables, including dictionaries, and the structure will be applied
        to the output from each datasets. For example:

            retrievable = {
                'my_id': 'ids',  # This converts to a Retrievable.
                'some_parameters': {
                    'x': 'grid/cell_size_x_si',  # This converts to a parameter.
                    't': 'time_step_size',  # This converts to a parameter.
                    },
                }

        could lead to an entry of output like

            {
                'my_id': 'asdfojlkadflkj',  # Some UUID.
                'some_parameters': {
                    'x': 1.0e-15,  # Looked up the nested parameter's value.
                    't': 1.7e-16,  # Looked up the top-level parameter's value.
                },
            }

        and similarly any non-dict-like iterables will result in lists with the corresponding information.

        What does the output contain?
        -----------------------------
        The content is always an iterable of information from none, 1 or more matching parameter sets.
        If `batch_size` is given, it will be an iterable of batches of those.

        Constraints can be
            - `ids`: Object ids of metadata sets (probably obtained from a separate call to `get`)
            - `parameters`: A dictionary of "parameter", `value` pairs
                            where value can either either denote an exact match
                            or a range if it is a slice.
            - `status`: A dictionary of "stage": "status" where "status" is a `Status`.

        How is the output ordered?
        --------------------------
        By default, the ordering is undefined and determined by the order in which the database provides the information.
        You can explicitly request a random ordering or provide a function of the parameter dictionary returning a float.
        In the latter case, the data will be sorted in ascending order with respect to these floats.

        Combining a clever ordering with a batch_size can produce interesting effects like "find the closest parameter set"
        (for a definition of "closest" as given by the function).
        """
        if not return_as_iterators:
            return _make_list(
                self.get(retrievable, return_as_iterators=True, batch_size=batch_size, **kwargs),
                0 if batch_size is None else 1,
            )
        return _get_batched_information(
            self.database, _retrievable_to_function(retrievable), batch_size=batch_size, **kwargs
        )
