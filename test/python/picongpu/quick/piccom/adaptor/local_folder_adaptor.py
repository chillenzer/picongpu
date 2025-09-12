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
