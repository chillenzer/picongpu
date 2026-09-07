"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from typing import Any

from pydantic import BaseModel, Field, model_serializer, model_validator

from .rendering import RenderedObject

JSON_SERIALISABLE_LEAVES = (bool, int, float, str)
"""
atomic values that survive a round trip through json.dumps / json.loads unaltered
"""


def check_rendering_context_is_json_serialisable(value: Any, path: str = "$") -> None:
    """
    ensure that every value reachable in the rendering context is JSON-serialisable

    Allows bool/int/float/str/None as leaves and list/dict as containers
    whose elements/values satisfy the same rule. Anything else (set, tuple,
    bytes, callables, arbitrary objects, ...) is rejected with a clear
    ValueError so that the mustache rendering context stays byte-deterministic.
    """
    if value is None or isinstance(value, JSON_SERIALISABLE_LEAVES):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            check_rendering_context_is_json_serialisable(item, f"{path}[{key!r}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            check_rendering_context_is_json_serialisable(item, f"{path}[{index}]")
        return
    raise ValueError(
        "custom user input values must be JSON-serialisable (bool/int/float/str/None "
        f"or list/dict thereof), but the value at {path} has type {type(value).__name__}: {value!r}"
    )


class CustomUserInput(RenderedObject, BaseModel):
    """
    container for easy passing of additional input as dict from user script to rendering context of simulation input
    """

    tags: list[str] | None = Field(None, min_length=1)
    """
    list of tags
    """

    rendering_context: dict[str, Any] | None = None
    """
    accumulation variable of added dictionaries
    """

    def check_does_not_change_existing_key_values(self, firstDict: dict, secondDict: dict):
        """check that updating firstDict with secondDict will not change any value in firstDict"""
        for key in firstDict.keys():
            if (key in secondDict) and (firstDict[key] != secondDict[key]):
                raise ValueError("Key " + str(key) + " exist already, and specified values differ.")

    def check_tags(self, existing_tags: list[str], tags: list[str]):
        """
        check that all entries in tags are valid tags and that all tags in the union if the list elements are unique
        """
        if "" in tags:
            raise ValueError("tags must not be empty string!")
        for tag in tags:
            if tag in existing_tags:
                raise ValueError("duplicate tag provided!, tags must be unique!")

    def addToCustomInput(self, custom_input: dict[str, Any], tag: str):
        """
        append dictionary to custom input dictionary
        """
        if tag == "":
            raise ValueError("tag must not be empty string!")
        if not custom_input:
            raise ValueError("custom input must contain at least 1 key")
        check_rendering_context_is_json_serialisable(custom_input)

        if (self.tags is None) and (self.rendering_context is None):
            self.tags = [tag]
            self.rendering_context = custom_input
        else:
            self.check_does_not_change_existing_key_values(self.rendering_context, custom_input)

            if tag in self.tags:
                raise ValueError("duplicate tag!")

            self.rendering_context.update(custom_input)
            self.tags.append(tag)

    def get_tags(self) -> list[str]:
        return self.tags

    @model_validator(mode="before")
    @classmethod
    def _parse_serialized(cls, value):
        # accept both the lossless per-entry form {"tags": ..., "rendering_context": ...}
        # produced by _get_serialized and the flat merged form (the "tags" list plus the
        # rendering context keys at the top level) as produced by Simulation's customuserinput
        # field serializer. The before-validator normalises the flat form into the per-entry
        # form so that the rendering context keys are not silently dropped (round-trip safety).
        if isinstance(value, dict) and "rendering_context" not in value:
            tags = value.get("tags")
            rendering_context = {key: entry for key, entry in value.items() if key != "tags"}
            return {"tags": tags, "rendering_context": rendering_context or None}
        return value

    @model_validator(mode="after")
    def _validate_json_serialisable_rendering_context(self):
        if self.rendering_context is not None:
            check_rendering_context_is_json_serialisable(self.rendering_context)
        return self

    @model_serializer(mode="plain")
    def _get_serialized(self) -> dict[str, Any] | None:
        # lossless form: carry both the tags and the rendering context so that the entry can
        # be reconstructed from its serialised form (round-trip safety); the *flattened*
        # (merged) form is produced by Simulation's customuserinput field serializer.
        if self.rendering_context is None and self.tags is None:
            return None
        return {"tags": self.tags, "rendering_context": self.rendering_context}
