from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class JointTransform(BaseModel):
    name: str = Field(min_length=1)
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


class JointRecipe(BaseModel):
    schema_version: Literal[1] = 1
    clean_mesh_artifact_id: str = Field(min_length=1)
    skeleton_template: str = Field(min_length=1)
    bone_naming: str = Field(min_length=1)
    mirror_enabled: bool = True
    transforms: list[JointTransform] = Field(min_length=1)
    editor_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_transforms(self) -> "JointRecipe":
        names = [item.name for item in self.transforms]
        if len(names) != len(set(names)):
            raise ValueError("Joint transform names must be unique")
        values = (
            value
            for item in self.transforms
            for value in (*item.position, *item.quaternion)
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Joint transforms must contain finite values")
        return self
