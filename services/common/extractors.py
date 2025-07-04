import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import more_itertools

from .files import FileHDF5
from .types import FeatureRecord, Frame
from .utils import get_logger

logger = get_logger("common.extractors")


class BaseExtractor:
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=8,
            help="number of frames to process in each slice of the dataset",
        )
        parser.add_argument(
            "--force",
            default=False,
            action="store_true",
            help="force re-extraction even if the output file already exists",
        )
        parser.add_argument(
            "--gpu",
            default=False,
            action="store_true",
            help="use GPU for extraction if available",
        )
        parser.add_argument(
            "--flush-interval",
            type=int,
            default=20,
            help="flush the output file every N frames processed",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        self.chunk_size: int = args.chunk_size
        self.force: bool = args.force
        self.gpu: bool = args.gpu
        self.flush_interval: int = args.flush_interval


class BaseFrameExtractor(BaseExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_dir",
            type=str,
            help="directory containing the frames to extract features from",
        )
        parser.add_argument(
            "-n",
            "--feature-name",
            type=str,
            required=True,
            help="name of the feature to extract (used hdf5 metadata)",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="path to the output file where features will be saved",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.frames_dir: Path = Path(args.frames_dir)
        self.feature_name: str = args.feature_name
        self.output_path: str = args.output

    def _load_frames(self) -> list[Frame]:
        if not self.frames_dir.is_dir():
            raise ValueError(
                f"Frames directory '{self.frames_dir}' does not exist or is not a directory"
            )
        frames: list[Frame] = []
        frame_paths = sorted(self.frames_dir.glob("*.jpg")) or sorted(
            self.frames_dir.glob("*.png")
        )
        for frame_path in frame_paths:
            frame = Frame(
                video_id=self.frames_dir.name, _id=frame_path.stem, path=frame_path
            )
            frames.append(frame)

        if not frames:
            raise ValueError(f"No frames found in directory '{self.frames_dir}'")

        return frames

    def _get_output_file(self, video_id: str, read_only: bool = False) -> FileHDF5:
        self.output_path.format(video_id=video_id)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        return FileHDF5(
            file_path=self.output_path,
            feature_name=self.feature_name,
            read_only=read_only,
            flush_interval=self.flush_interval,
        )

    def extract_list(self, frame_paths: list[Path]) -> list[FeatureRecord]:
        raise NotImplementedError("This method should be implemented in subclasses")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[FeatureRecord]:
        assert self.chunk_size > 0, "Chunk size must be greater than 0"

        batched_frames = more_itertools.chunked(frame_paths, self.chunk_size)
        batched_records = map(self.extract_list, batched_frames)
        records = itertools.chain.from_iterable(batched_records)
        yield from records

    def _skip_extracted_frames(self, frames: list[Frame]) -> Iterator[Frame]:
        frame_groups = sorted(
            ((frame.video_id, frame._id, frame.path) for frame in frames),
            key=lambda x: x[0],
        )
        for video_id, group in itertools.groupby(frame_groups, key=lambda x: x[0]):
            with self._get_output_file(video_id, read_only=True) as file:
                skipping_frames: list[Frame] = []
                for video_id, frame_id, frame_path in group:
                    if frame_id not in file:
                        skipping_frames.append(
                            Frame(video_id=video_id, _id=frame_id, path=frame_path)
                        )
            logger.info(
                f"Skipping {len(frames) - len(skipping_frames)} frames for video '{video_id}'"
            )
            yield from skipping_frames

    def run(self) -> None:
        frames = self._load_frames()

        if not self.force and Path(self.output_path).exists():
            frames = list(self._skip_extracted_frames(frames))

        if not frames:
            logger.info("Exiting because all frames are already extracted")
            return

        # Unpack frames into video IDs, frame IDs, and paths
        video_ids, frame_ids, frame_paths = zip(
            *[(frame.video_id, frame._id, frame.path) for frame in frames]
        )
        records = self.extract_iterable(frame_paths)

        # Group records by video ID
        triples = zip(video_ids, frame_ids, records)
        video_groups: dict[str, list[tuple[str, FeatureRecord]]] = defaultdict(list)
        for video_id, frame_id, record in triples:
            video_groups[video_id].append((frame_id, record))

        num_videos = len(video_groups)
        num_records = sum(len(items) for items in video_groups.values())

        for video_id, items in video_groups.items():
            feature_records = [
                FeatureRecord(_id=frame_id, feature=record.feature)
                for frame_id, record in items
            ]
            with self._get_output_file(video_id, read_only=False) as file:
                file.save_all(feature_records, force=self.force)

        logger.info(f"Saved features to {Path(self.output_path).as_posix()}")
        logger.info(f"Extracted {num_records} features from {num_videos} videos")


if __name__ == "__main__":
    pass
