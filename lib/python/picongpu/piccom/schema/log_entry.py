"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel


class LogEntry(BaseModel):
    action_name: str
    update_of: str | None = None
    timestamp: datetime
    content: Any
