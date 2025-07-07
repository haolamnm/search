from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeatureRecord:
    _id: str
    feature_vector: list[float]


@dataclass
class ObjectRecord:
    _id: str
    detector: str
    scores: list[float] | None = None
    yxyx_boxes: list[tuple[float, float, float, float]] | None = None
    labels: list[int] | None = None
    entities: list[str] | None = None
    names: list[str] | None = None
    monochrome: float | None = None
    cluster_code: str | None = None
    feature_str: str | None = None


@dataclass
class Frame:
    video_id: str
    _id: str
    path: Path


@dataclass
class Scene:
    video_id: str
    _id: str
    video_path: Path
    start_frame: int
    start_time: float
    end_frame: int
    end_time: float


if __name__ == "__main__":
    pass
