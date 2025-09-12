"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any

from picongpu.piccom.schema.metadata_file import MetadataFile


def _contains(content, parameters):
    return all(key in content and content[key] == value for key, value in parameters.items())


def _extract_parameters(content: MetadataFile | dict) -> dict[str, Any]:
    if isinstance(content, dict):
        return _extract_parameters(MetadataFile.model_validate(content))
    return next(filter(lambda x: x.action_name == "generate_input_files", content.log.values())).content


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
