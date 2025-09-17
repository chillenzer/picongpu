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
from picongpu.piccom.adaptor.local_folder_adaptor import Parameter, _extract_parameters
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
    statuses = cycle(["success", "failure", None, "failure", "success"])
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


def _flatten(my_list):
    return sum(my_list, [])


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
        self.assertSetEqual(set(ids), set(self.adaptor.get("ids")))

    def test_handles_exact_parameter_matches(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        arbitrary_id, arbitrary_content = next(iter(objects.items()))
        self.assertSequenceEqual(self.adaptor.get("ids", parameters=arbitrary_content), [arbitrary_id])

    def test_returns_empty_list_if_no_match(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        arbitrary_content = {"p": "asdf"}
        self.assertSequenceEqual(self.adaptor.get("ids", parameters=arbitrary_content), [])

    def test_handles_multiple_parameter_matches(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        arbitrary_content = next(iter(objects.values()))
        arbitrary_content.pop("y")
        found_ids = self.adaptor.get("ids", parameters=arbitrary_content)

        # It felt somehow most straightforward to test set equality by inclusion in both directions:
        for i in found_ids:
            self.assertEqual(_extract_parameters(self.database.get_content(i))["x"], arbitrary_content["x"])
        for i in set(objects.keys()) - set(found_ids):
            self.assertNotEqual(_extract_parameters(self.database.get_content(i))["x"], arbitrary_content["x"])

    def test_handles_slices(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])

        for s in [slice(0, 2), slice(4), slice(1, None), slice(None)]:
            arbitrary_slice = {"x": s}
            found_ids = self.adaptor.get("ids", parameters=arbitrary_slice)

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
            result = self.adaptor.get("ids", status={"generate_input_files": s})
            self.assertSetEqual(set(result), set(map(lambda x: x[0], status_ids(ids, s))))

    def test_filters_by_status_on_another_action(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        populate_updates(self.database, {"z": 123}, action="build")
        ids = add_status(self.database)
        for s in ["success", "failure", "started", "ended", "running"]:
            result = self.adaptor.get("ids", status={"build": s})
            self.assertSetEqual(set(result), set(map(lambda x: x[0], status_ids(ids, s, "build"))))

    def test_filters_by_status_on_multiple_actions(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        populate_updates(self.database, {"z": 123}, action="build")
        ids = add_status(self.database)
        for s1 in ["success", "failure", "started", "ended", "running"]:
            for s2 in ["success", "failure", "started", "ended", "running"]:
                result = set(self.adaptor.get("ids", status={"generate_input_files": s1, "build": s2}))
                expected = set(map(lambda x: x[0], status_ids(ids, s1, "generate_input_files"))).intersection(
                    set(map(lambda x: x[0], status_ids(ids, s2, "build")))
                )
                self.assertSetEqual(result, expected)

    def test_filters_by_status_started(self):
        populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        add_status(self.database)
        result = self.adaptor.get("ids", status={"non-existent-stage": "started"})
        self.assertSetEqual(set(result), set())

    def test_get_parameter_sets(self):
        objects = populate(self.database, [{"x": x, "y": y} for x in range(7) for y in range(42, 49)])
        # we add a bit of noise here in order to ensure that the algorithm can handle that:
        populate_updates(self.database, {"z": 123}, action="build")
        add_status(self.database)

        result = self.adaptor.get("parameters")

        # dictionaries are not hashable, so we can't use a set comparison here
        for o in objects.values():
            self.assertTrue(o in result)
        for o in result:
            self.assertTrue(o in objects.values())

    def test_batch_size(self):
        objects = populate(self.database)
        for batch_size in range(1, len(objects) + 3):
            result = self.adaptor.get("ids", batch_size=batch_size)
            self.assertEqual(len(_flatten(result)), len(objects))
            for batch in result[:-1]:
                self.assertEqual(len(batch), batch_size)
            self.assertLessEqual(len(result[-1]), batch_size)

    def test_ordering(self):
        populate(self.database)
        results = [self.adaptor.get("ids", ordering="random") for _ in range(10)]
        # all sets of IDs are identical, a few hoops to jump through to find a hashable type
        self.assertEqual(len(set(map(tuple, map(sorted, results)))), 1)
        # they are ordered differently (up to a very small chance of bad luch here)
        self.assertGreater(len(set(map(tuple, results))), 1)

    def test_get_by_ids(self):
        ids = populate(self.database).keys()
        for i in ids:
            self.assertEqual(self.adaptor.get("ids", ids=[i]), [i])
        for i1 in ids:
            for i2 in ids:
                self.assertEqual(set(self.adaptor.get("ids", ids=[i1, i2])), {i1, i2})

    def test_get_parameter_value(self):
        objects = populate(self.database, [{"x": x, "y": {"z": y, "a": "b"}} for x in range(7) for y in range(42, 49)])
        for i, content in objects.items():
            self.assertEqual(self.adaptor.get("x", ids=[i])[0], content["x"])
            self.assertEqual(self.adaptor.get(Parameter("x"), ids=[i])[0], content["x"])
            self.assertEqual(self.adaptor.get(Parameter("y"), ids=[i])[0], content["y"])
            self.assertEqual(self.adaptor.get(Parameter("y/z"), ids=[i])[0], content["y"]["z"])
            self.assertEqual(self.adaptor.get(["x", "y"], ids=[i])[0], [content["x"], content["y"]])
            self.assertEqual(self.adaptor.get(["ids", "y/a"], ids=[i])[0], [i, content["y"]["a"]])
            self.assertEqual(
                self.adaptor.get({"id": "ids", "c": {"a": "y/a", "l": ["x", "y/z"]}}, ids=[i])[0],
                {"id": i, "c": {"a": "b", "l": [content["x"], content["y"]["z"]]}},
            )

    def test_ordering_by_metric(self):
        populate(self.database, [{"x": x, "y": {"z": y, "a": "b"}} for x in range(7) for y in range(42, 49)])

        # This is sorted by the metric:
        result = self.adaptor.get("y/z", ordering=lambda par: par["y"]["z"])
        self.assertEqual(result, sorted(result))

        # just to be sure: This is not.
        result = self.adaptor.get("y/z")
        self.assertNotEqual(result, sorted(result))
