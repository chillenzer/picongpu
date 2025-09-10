"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from uuid import uuid4 as uuid
from os import PathLike
from pathlib import Path


def interpret_dot_notation(spec):
    if len(spec) == 1 and "." not in next(iter(spec)):
        return spec
    return {
        (split_key := key.split(".", 1))[0]: interpret_dot_notation({split_key[1]: value})
        for key, value in spec.items()
    }


def merge_into(into_dict, from_dict):
    if not isinstance(from_dict, dict):
        return from_dict
    return into_dict | {key: merge_into(into_dict.get(key, {}), value) for key, value in from_dict.items()}


class LocalFolderDatabase:
    """
    Simple mongoDB-like database storing json files on disk
    """

    def __init__(self, directory: PathLike):
        self.directory = Path(directory)
        if self.directory.exists() and not self.directory.is_dir():
            raise ValueError("{directory=} should point to a directory usable for storage.")

    def __getitem__(self, collection):
        return LocalFolderDatabase(self.directory / collection)

    def _generate_id(self, content):
        return uuid().hex

    def insert_one(self, content: dict, identifier=None):
        if identifier is None:
            return self.insert_one(content, self._generate_id(content))
        content_with_id = {"_id": identifier} | content
        path = self.get_path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(content_with_id, file)
        return content_with_id

    def get_path(self, identifier):
        return self.directory / f"{identifier}.json"

    def get_content(self, identifier):
        path = self.get_path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("r") as file:
            return json.load(file)

    def get_directory(self):
        return self.directory

    def update_one(self, identifier, operation):
        if isinstance(identifier, dict):
            return self.update_one(identifier["_id"], operation)
        if unknown_operations := set(operation.keys()) - {"$set"}:
            raise NotImplementedError(
                f"You have tried to update with operations {unknown_operations}. This is not yet implemented."
            )
        return self.insert_one(
            merge_into(self.get_content(identifier), interpret_dot_notation(operation["$set"])), identifier
        )
