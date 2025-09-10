"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from datetime import datetime
from typing import Any, Literal, Iterable

from pydantic import BaseModel


class MetadataFile(BaseModel):
    username: str
    date_time: datetime
    log: Any
    upload_type: Literal["PIConGPU"] = "PIConGPU"
    keywords: Iterable[str] = tuple()
    description: str = ""
