"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from tempfile import TemporaryDirectory
from picongpu.piccom.db import LocalFolderDatabase


class Communicator(LocalFolderDatabase):
    def __init__(self, author, *args, **kwargs):
        self.author = author
        if len(args) == 0 and "directory" not in kwargs:
            kwargs["directory"] = TemporaryDirectory(delete=False).name
        super().__init__(*args, **kwargs)

    def insert_one(self, content: dict, identifier: str | None = None):
        return super().insert_one({"username": self.author} | content, identifier)

    def print_info(self):
        logging.info(f"Local folder database created in {self.directory=}.")
