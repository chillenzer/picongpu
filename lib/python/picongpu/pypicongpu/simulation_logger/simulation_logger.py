"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from collections import namedtuple
from functools import singledispatch
import json
from datetime import datetime, timezone
from typing import Callable, Any
from uuid import uuid4 as uuid

METADATA_FORMAT_VERSION = "0.1.0"


@singledispatch
def make_serialisable(obj):
    try:
        # maybe it is already serialisable?
        json.dumps(obj)
        return obj
    except TypeError:
        # don't panic, we've got more options at our disposal
        pass

    try:
        # maybe it knows how to serialise itself?
        return obj.make_serialisable()
    except AttributeError as e:
        raise ValueError(f"I don't know how to make {obj} of type {type(obj)} serialisable.") from e


def _generate_action_id(payload):
    return uuid().hex


def _compose_payload(serialisable_object=None, action_name=None, update_of=None):
    return {
        "action_name": action_name or "unspecified",
        "update_of": update_of,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": serialisable_object or {},
    }


def _get_id(obj):
    return obj["_id"]


class SimulationLogger:
    def __init__(self, database):
        self.database = database
        self.action_id_to_object = dict()

    def log(self, payload, action=None):
        payload = _compose_payload(make_serialisable(payload), action_name=action)
        action_id = _generate_action_id(payload)
        object_id = _get_id(
            self.database.insert_one(
                {
                    "log": {action_id: payload},
                    "metadata_format_version": METADATA_FORMAT_VERSION,
                    "upload_type": "PIConGPU",
                    "date_time": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        self.action_id_to_object[action_id] = object_id
        return action_id

    def success(self, action_id, info=None):
        return self.update(action_id, action="success", payload={"status": "success", "info": info})

    def failure(self, action_id, exception=None, info=None):
        return self.update(
            action_id,
            action="failure",
            payload={"status": "exception" if exception else "failure", "info": info}
            | ({"exception": str(exception)} if exception else {}),
        )

    def update(self, action_id, action=None, payload=None):
        payload = _compose_payload(serialisable_object=payload, action_name=action, update_of=action_id)
        new_action_id = _generate_action_id(payload)
        object_id = self.get_object_id(action_id)
        self.database.update_one({"_id": object_id}, {"$set": {f"log.{new_action_id}": payload}})
        self.action_id_to_object[new_action_id] = object_id
        return new_action_id

    def get_object_id(self, action_id):
        return self.action_id_to_object[action_id]

    def get_full_content(self, identifier):
        return self.database.get_content(self.action_id_to_object.get(identifier, identifier))

    def get_log(self, action_id):
        return self.get_full_content(action_id)["log"]

    def get_action(self, action_id):
        return self.get_full_content(action_id)["log"][action_id]

    def get_all_objects(self):
        return tuple(set(self.action_id_to_object.values()))


def logged_operation(
    func=None,
    /,
    action=None,
    capture_args=True,
    capture_result=True,
    info: Callable[[Any], dict[str, Any]] | dict[str, Any] | None = None,
):
    info = info or {}
    if isinstance(info, dict):
        return logged_operation(
            func, action=action, capture_args=capture_args, capture_result=capture_result, info=lambda _: info
        )
    if func is not None:
        return logged_operation(
            action=action, capture_args=capture_args, capture_result=capture_result, info=lambda _: info
        )(func)

    def decorator(func):
        def tmp(self, *args, return_action_id=False, update_of=None, **kwargs):
            local_info = info(self) | ({"args": args, "kwargs": kwargs} if capture_args else {})
            if update_of is not None:
                action_id = self.logger.update(action_id=update_of, action=action, payload={"info": local_info})
            else:
                payload = make_serialisable(self)
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                payload = {"info": local_info} | payload

                action_id = self.logger.log(payload=payload, action=action)
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                self.logger.failure(action_id, exception=e)
                raise
            else:
                self.logger.success(action_id, info={"result": result if capture_result else None})
            if return_action_id:
                LoggedOperationResult = namedtuple("LoggedOperationResult", ["result", "action_id"])
                return LoggedOperationResult(result, action_id)
            return result

        return tmp

    return decorator
