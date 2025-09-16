"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from datetime import datetime, timezone
from logging import warning
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from itertools import cycle

import numpy as np
from picongpu.piccom.adaptor import LocalFolderAdaptor
from picongpu.piccom.adaptor.local_folder_adaptor import _extract_parameters
from picongpu.piccom.db import LocalFolderDatabase
from picongpu.piccom.schema import LogEntry, MetadataFile


def is_empty(path):
    if not isinstance(path, Path):
        return is_empty(Path(path))
    if not path.is_dir():
        raise ValueError(f"You've asked if {path} is empty but it is not a directory.")
    return not any(path.iterdir())


def populate(database, parameters=None):
    log_entries = {f"uuid{i}": p for i, p in enumerate(parameters or range(10))}
    return {
        database.insert_one(
            MetadataFile(
                username="unimportant",
                date_time=datetime.now(timezone.utc),
                log={
                    u: LogEntry(action_name="generate_input_files", timestamp=datetime.now(timezone.utc), content=entry)
                },
            )
        )["_id"]: entry
        for u, entry in log_entries.items()
    }


def populate_updates(database, parameters, action):
    result = []
    for obj in database.find():
        my_uuid = next(iter(obj["log"].keys()))
        result.append(
            database.update_one(
                obj["_id"],
                {
                    "$set": {
                        f"log.{my_uuid}a": {
                            "update_of": my_uuid,
                            "action_name": action,
                            "timestamp": datetime.now(timezone.utc),
                            "content": parameters,
                        }
                    }
                },
            )
        )
    return result


def events_in(database):
    return sum(map(lambda obj: list(map(lambda x: (obj["_id"],) + x, obj["log"].items())), database.find()), [])


def add_status(database):
    statuses = cycle(["success", "failure", None])
    ids = {}
    for (obj_id, my_uuid, obj), s in zip(events_in(database), statuses):
        ids.setdefault(s, []).append((obj_id, obj["action_name"], my_uuid))
        if s is not None:
            database.update_one(
                obj_id,
                {
                    "$set": {
                        f"log.{my_uuid}s": {
                            "update_of": my_uuid,
                            "action_name": s,
                            "timestamp": datetime.now(timezone.utc),
                            "content": {},
                        }
                    }
                },
            )
    return ids


def status_ids(ids, status, action=None):
    result = None
    if status == "success":
        result = ids["success"]
    if status == "failure":
        result = ids["failure"]
    if status == "ended":
        result = ids["success"] + ids["failure"]
    if status == "started":
        result = ids["success"] + ids["failure"] + ids[None]
    if status == "running":
        result = ids[None]
    if result is None:
        raise ValueError(f"Unknown {status=} requested.")

    if action is not None:
        result = list(filter(lambda x: x[1] == action, result))

    return result


class TestLocalFolderAdaptor(TestCase):
    def setUp(self) -> None:
        try:
            # On POSIX-compliant linux systems, this would reside in memory.
            # Good for testing because we don't have to go to the physical disc.
            self.storage = TemporaryDirectory(dir="/dev/shm")
        except Exception:
            # Not sure what to catch here... sorry!
            # We'll be a bit slower with this:
            warning("Using in-memory directory failed.")
            self.storage = TemporaryDirectory()

        self.my_dir = Path(self.storage.name)
        self.assertTrue(is_empty(self.my_dir))
        self.database = LocalFolderDatabase(self.my_dir)
        self.adaptor = LocalFolderAdaptor(self.database)

    def tearDown(self) -> None:
        self.storage.cleanup()

    def test_gets_all_ids(self):
        ids = populate(self.database).keys()
        self.assertSetEqual(set(ids), set(self.adaptor.get_ids()))

    def test_handles_exact_parameter_matches(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        arbitrary_id, arbitrary_content = next(iter(objects.items()))
        self.assertSequenceEqual(self.adaptor.get_ids(parameters=arbitrary_content), [arbitrary_id])

    def test_returns_empty_list_if_no_match(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        arbitrary_content = {"p": "asdf"}
        self.assertSequenceEqual(self.adaptor.get_ids(parameters=arbitrary_content), [])

    def test_handles_multiple_parameter_matches(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        arbitrary_content = next(iter(objects.values()))
        arbitrary_content.pop("y")
        found_ids = self.adaptor.get_ids(parameters=arbitrary_content)

        # It felt somehow most straightforward to test set equality by inclusion in both directions:
        for i in found_ids:
            self.assertEqual(_extract_parameters(self.database.get_content(i))["x"], arbitrary_content["x"])
        for i in set(objects.keys()) - set(found_ids):
            self.assertNotEqual(_extract_parameters(self.database.get_content(i))["x"], arbitrary_content["x"])

    def test_handles_slices(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])

        for s in [slice(0, 2), slice(4), slice(1, None), slice(None)]:
            arbitrary_slice = {"x": s}
            found_ids = self.adaptor.get_ids(parameters=arbitrary_slice)

            # It felt somehow most straightforward to test set equality by inclusion in both directions:
            for i in objects.keys():
                content = _extract_parameters(self.database.get_content(i))
                if i in found_ids:
                    self.assertLessEqual(content["x"], arbitrary_slice["x"].stop or np.inf)
                    self.assertGreaterEqual(content["x"], arbitrary_slice["x"].start or -np.inf)
                else:
                    if "x" in content:
                        self.assertGreater(content["x"], arbitrary_slice["x"].stop or -np.inf)
                        self.assertLess(content["x"], arbitrary_slice["x"].start or np.inf)

    def test_filters_by_status(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        ids = add_status(self.database)
        for s in ["success", "failure", "started", "ended", "running"]:
            result = self.adaptor.get_ids(status={"generate_input_files": s})
            self.assertSetEqual(set(result), set(map(lambda x: x[0], status_ids(ids, s))))

    def test_filters_by_status_on_another_action(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        populate_updates(self.database, [{"z": z} for z in range(123, 127)], action="build")
        ids = add_status(self.database)
        for s in ["success", "failure", "started", "ended", "running"]:
            result = self.adaptor.get_ids(status={"build": s})
            self.assertSetEqual(set(result), set(map(lambda x: x[0], status_ids(ids, s, "build"))))
