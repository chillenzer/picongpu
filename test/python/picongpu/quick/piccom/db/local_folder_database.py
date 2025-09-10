"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
import tempfile
import unittest
from pathlib import Path

from picongpu.piccom.db import LocalFolderDatabase


def is_empty(path):
    if not isinstance(path, Path):
        return is_empty(Path(path))
    if not path.is_dir():
        raise ValueError(f"You've asked if {path} is empty but it is not a directory.")
    return not any(path.iterdir())


class TestLocalFolderDatabase(unittest.TestCase):
    def setUp(self) -> None:
        try:
            # On POSIX-compliant linux systems, this would reside in memory.
            # Good for testing because we don't have to go to the physical disc.
            self.storage = tempfile.TemporaryDirectory(dir="/dev/shm")
        except Exception:
            # Not sure what to catch here... sorry!
            # We'll be a bit slower with this:
            logging.warning("Using in-memory directory failed.")
            self.storage = tempfile.TemporaryDirectory()

        self.my_dir = Path(self.storage.name)
        self.assertTrue(is_empty(self.my_dir))
        self.database = LocalFolderDatabase("testauthor", self.my_dir)

    def tearDown(self) -> None:
        self.storage.cleanup()

    def test_insert_one_generates_file(self):
        self.database.insert_one(dict())
        self.assertFalse(is_empty(self.my_dir))

    def test_can_tell_the_path_of_a_file(self):
        identifier = self.database.insert_one(dict())["_id"]
        path = self.database.get_path(identifier)
        self.assertTrue(path.exists())

    def test_returns_a_path_inside_of_directory(self):
        identifier = self.database.insert_one(dict())["_id"]
        path = self.database.get_path(identifier)
        self.assertTrue(self.my_dir.absolute() in path.absolute().parents)

    def test_returns_content_of_identifier(self):
        data = dict(x=1, y="asdf", z=False)
        identifier = self.database.insert_one(data)["_id"]
        content = self.database.get_content(identifier)
        self.assertEqual({"_id": identifier} | data, content)

    def test_interprets_string_as_path(self):
        self.assertTrue(isinstance(LocalFolderDatabase("testauthor", "abc").get_directory(), Path))

    def test_updates_with_set_operator(self):
        arbitrary_value = 7
        identifier = self.database.insert_one(dict())["_id"]
        self.database.update_one(identifier, {"$set": {"x": arbitrary_value}})
        self.assertEqual(self.database.get_content(identifier)["x"], arbitrary_value)

    def test_updates_with_id_in_dict(self):
        arbitrary_value = 7
        identifier = self.database.insert_one(dict())["_id"]
        self.database.update_one({"_id": identifier}, {"$set": {"x": arbitrary_value}})
        self.assertEqual(self.database.get_content(identifier)["x"], arbitrary_value)

    def test_set_operator_understands_dot_notation(self):
        arbitrary_value = 7
        identifier = self.database.insert_one(dict(x={"y": 42}))["_id"]
        self.database.update_one({"_id": identifier}, {"$set": {"x.y": arbitrary_value}})
        self.assertEqual(self.database.get_content(identifier)["x"]["y"], arbitrary_value)

    def test_set_operator_preserves_other_content_with_dot_notation(self):
        arbitrary_value = 7
        identifier = self.database.insert_one(dict(x={"y": 42, "z": arbitrary_value}))["_id"]
        self.database.update_one({"_id": identifier}, {"$set": {"x.y": 4}})
        self.assertEqual(self.database.get_content(identifier)["x"]["z"], arbitrary_value)

    def test_update_raises_for_any_other_operation_than_set(self):
        arbitrary_value = 7
        identifier = self.database.insert_one(dict(x={"y": 42}))["_id"]
        message = r"You have tried to update with operations {'\$pull'}. This is not yet implemented."
        with self.assertRaisesRegex(NotImplementedError, message):
            self.database.update_one({"_id": identifier}, {"$pull": {"x.y": arbitrary_value}})

    def test_creates_directories_as_needed(self):
        database = LocalFolderDatabase("testauthor", self.my_dir / "this" / "does" / "not" / "exist")
        identifier = database.insert_one(dict())["_id"]
        self.assertEqual(database.get_content(identifier)["_id"], identifier)


if __name__ == "__main__":
    unittest.main()
