"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any

import numpy as np

from picongpu.piccom.schema.metadata_file import MetadataFile


def _contains_single_value(lhs, rhs):
    if isinstance(rhs, slice):
        return lhs >= (rhs.start or -np.inf) and lhs <= (rhs.stop or np.inf)
    return lhs == rhs


def _contains(content, parameters):
    return all(key in content and _contains_single_value(content[key], value) for key, value in parameters.items())


ACTION_NAME = {"parameters": "generate_input_files", "runtime_info": "run"}


def _extract(content: MetadataFile | dict, what) -> dict[str, Any]:
    if isinstance(content, dict):
        return _extract_parameters(MetadataFile.model_validate(content))
    return next(filter(lambda x: x.action_name == ACTION_NAME[what], content.log.values())).content


def _extract_parameters(content: MetadataFile | dict) -> dict[str, Any]:
    return _extract(content, "parameters")


class LocalFolderAdaptor:
    def __init__(self, database):
        self.database = database

    def get_ids(self, parameters: dict[str, Any] | None = None):
        all_ids = [obj["_id"] for obj in self.database.find()]
        if parameters is None:
            return all_ids

        return [
            i
            for i, content in zip(all_ids, map(lambda i: _extract_parameters(self.database.get_content(i)), all_ids))
            if _contains(content, parameters)
        ]
