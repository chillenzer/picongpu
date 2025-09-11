"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from datetime import datetime
from typing import Literal, Iterable

from picongpu.piccom.schema.log_entry import LogEntry
from picongpu.piccom.schema.version import METADATA_FORMAT_VERSION, AnyMetadataFormatVersion
from pydantic import BaseModel


class MetadataFile(BaseModel):
    username: str
    date_time: datetime
    log: dict[str, LogEntry]
    metadata_format_version: AnyMetadataFormatVersion = METADATA_FORMAT_VERSION
    upload_type: Literal["PIConGPU"] = "PIConGPU"
    # pydantic warns if this is a standard type like [] or tuple()
    # using those in production should be fine
    keywords: Iterable[str] = iter([])
    description: str = ""
