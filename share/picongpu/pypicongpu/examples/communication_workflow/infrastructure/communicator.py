"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from copy import deepcopy
from datetime import datetime, timezone
from picongpu.piccom.communicator import Communicator
from picongpu.piccom.schema import LogEntry
from picongpu.pypicongpu.simulation_logger.simulation_logger import _generate_action_id

from .parameters import DIRECTORIES


class MyCommunicator(Communicator):
    additional_content = None

    def record_additionally(self, content):
        self.additional_content = content

    def clear_record(self):
        self.additional_content = None

    def insert_one(self, content: dict, identifier: str | None = None):
        original_return = super().insert_one(content, identifier)
        if self.additional_content is not None:
            parent_action = next(
                filter(
                    lambda obj: obj[1]["action_name"] == "generate_input_files",
                    original_return["log"].items(),
                )
            )[0]
            payload = deepcopy(self.additional_content)
            # Doing this here is necessary because we'd otherwise recurse indefinitely:
            self.clear_record()
            self.update_one(
                {"_id": original_return["_id"]},
                {
                    "$set": {
                        f"log.{_generate_action_id(payload)}": LogEntry(
                            action_name="additional_parameters",
                            update_of=parent_action,
                            timestamp=datetime.now(timezone.utc),
                            content=payload,
                        ).model_dump(mode="json")
                    }
                },
            )
        return original_return


COMMUNICATOR = MyCommunicator(author="Julian Lenz", directory=DIRECTORIES["database"]())
