"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import reduce
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Iterable, Literal
from hashlib import sha256

import tomli_w
from pydantic import AfterValidator, BaseModel

from picongpu.pypicongpu.output.plugin import Plugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species.species import Species


class OpenPMDConfig(BaseModel):
    file: PathLike | str
    infix: str = "_%06T"
    ext: Annotated[str, AfterValidator(lambda s: s.strip("."))] = "h5"
    backend_config: PathLike | None = None
    data_preparation_strategy: Literal["mappedMemory", "doubleBuffer"] = "mappedMemory"
    range: None = None

    def full_filename(self):
        return f"{self.file}{self.infix}.{self.ext}"

    def result_path(self, prefix_path: PathLike = Path()):
        filename = self.full_filename()
        if Path(filename).is_absolute():
            return filename
        return (Path(prefix_path) / filename).absolute()


def to_string(timestepspec: TimeStepSpec):
    return ",".join(
        map(
            lambda x: "{start}:{stop}:{step}".format(**x),
            timestepspec.get_rendering_context()["specs"],
        )
    )


class FieldDump(BaseModel):
    name: str

    def get_rendering_context(self) -> dict:
        return self.model_dump(mode="json")


class OpenPMDPlugin(Plugin):
    _name = "openPMD"

    sources: list[tuple[TimeStepSpec, Iterable[Species]]]
    config: OpenPMDConfig = OpenPMDConfig(file="simData")
    _setup_dir: Path | None = None
    # We're using a negation here because now `False` and `None` (evaluating to `False`)
    # both mean that we can't rely on `setup_dir` being anything permanent:
    _setup_dir_is_not_temporary: bool | None = None

    def config_filename(self, content, context: Literal["runtime", "setup"]):
        filename = f"openPMD_config_{sha256(tomli_w.dumps(content).encode()).hexdigest()}.toml"
        if not self._setup_dir_is_not_temporary or context == "setup":
            return self.setup_dir / "etc" / filename
        if context == "runtime":
            return Path("..") / "input" / "etc" / filename
        raise ValueError(f"Unknown {context=} upon requesting the openPMD config filename.")

    @property
    def setup_dir(self):
        if self._setup_dir_is_not_temporary is None:
            self._setup_dir_is_not_temporary = self._setup_dir is not None

        if self._setup_dir is None:
            self._setup_dir = Path(TemporaryDirectory(delete=False).name).absolute()

        return self._setup_dir

    @setup_dir.setter
    def setup_dir(self, other):
        self._setup_dir = Path(other)

    def __init__(self, sources, config):
        self.sources = sources
        self.config = config

    def _generate_config_file(self):
        # There's some strange interaction with the custom hashing of TimeStepSpec
        # that's implemented on RenderedObject
        # hindering the storage of this data structure.
        # As a workaround, we're computing this on the fly.
        # Shouldn't be performance critical but it would be more elegant to normalise early on.
        sources = reduce(
            lambda dictionary, key_val: dictionary.setdefault(to_string(key_val[0]), []).append(
                key_val[1].get_rendering_context()["name"]
            )
            or dictionary,
            self.sources,
            {},
        )
        content = self.config.model_dump(mode="json", exclude_none=True) | {
            "sink": {"dummy_application_name": {"period": sources}}
        }
        with self.config_filename(content, context="setup").open("wb") as file:
            tomli_w.dump(content, file)
        return content

    def _get_serialized(self) -> dict | None:
        content = self._generate_config_file()
        return {"config_filename": str(self.config_filename(content, context="runtime"))}

    class Config:
        arbitrary_types_allowed = True
