"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picongpu.piccom.db import LocalFolderDatabase


class Communicator(LocalFolderDatabase):
    def __init__(self, author, *args, **kwargs):
        self.author = author
        super().__init__(*args, **kwargs)

    def print_info(self):
        print(f"Local folder database created in {self.directory=}.")
