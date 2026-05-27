from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import dataclasses
import io
import logging
from pathlib import Path
from typing import Any, Generic, TypeVar

import av
from google.protobuf import message as protobuf_message
from google.protobuf import timestamp_pb2
from mcap.reader import make_reader
import mcap_protobuf.writer
import numpy as np

from openpi.pistream.proto import descriptor_pb2
from openpi.pistream.proto import encoding_pb2
from openpi.pistream.proto import frame_pb2

PI_STREAM_FRAME_PREFIX = "frame/"
PI_STREAM_DESCRIPTOR_TOPIC = "stream_descriptor"
MCAP_CHUNK_SIZE = 1024 * 1024

_UserdataT = TypeVar("_UserdataT")


@dataclasses.dataclass(frozen=True, order=True)
class FieldKey:
    publisher_id: str
    key: str


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    publisher_id: str
    key: str
    encoding: encoding_pb2.Encoding
    message_type: type[protobuf_message.Message]
    shard: int = 0

    @property
    def field_key(self) -> FieldKey:
        return FieldKey(self.publisher_id, self.key)

    @property
    def channel_topic(self) -> str:
        return f"{PI_STREAM_FRAME_PREFIX}{self.publisher_id}/{self.key}"

    def to_descriptor_proto(self) -> descriptor_pb2.FieldDescriptor:
        return descriptor_pb2.FieldDescriptor(
            publisher_id=self.publisher_id,
            key=self.key,
            encoding=self.encoding,
        )


def timestamp_from_nanos(timestamp_ns: int) -> timestamp_pb2.Timestamp:
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromNanoseconds(int(timestamp_ns))
    return timestamp


def timestamp_to_nanos(timestamp: timestamp_pb2.Timestamp) -> int:
    return int(timestamp.seconds) * 1_000_000_000 + int(timestamp.nanos)


def ndarray_encoding(shape: tuple[int, ...], dtype: np.dtype | str) -> encoding_pb2.Encoding:
    return encoding_pb2.Encoding(
        ndarray=encoding_pb2.NDArrayContent(
            shape=list(shape),
            dtype=np.dtype(dtype).name,
            default_encoding=encoding_pb2.DefaultEncoding(),
        )
    )


def string_encoding() -> encoding_pb2.Encoding:
    return encoding_pb2.Encoding(
        string_content=encoding_pb2.StringContent(default_encoding=encoding_pb2.DefaultEncoding())
    )


def timestamp_encoding() -> encoding_pb2.Encoding:
    return encoding_pb2.Encoding(
        timestamp=encoding_pb2.TimestampContent(default_encoding=encoding_pb2.DefaultEncoding())
    )


def snapshot_ref_encoding(fields: Iterable[FieldKey]) -> encoding_pb2.Encoding:
    field_index_map = [
        encoding_pb2.PiStreamSnapshotRefContent.FieldIndexMapEntry(
            publisher_id=field.publisher_id,
            key=field.key,
            index=index,
        )
        for index, field in enumerate(sorted(set(fields), key=lambda field: (field.key, field.publisher_id)), start=1)
    ]
    return encoding_pb2.Encoding(
        snapshot_ref=encoding_pb2.PiStreamSnapshotRefContent(
            field_index_map=field_index_map,
            default_encoding=encoding_pb2.DefaultEncoding(),
        )
    )


def compressed_video_encoding(*, width: int, height: int, fps: float) -> encoding_pb2.Encoding:
    return encoding_pb2.Encoding(
        video=encoding_pb2.VideoContent(
            width=width,
            height=height,
            pix_fmt="yuv420p",
            compressed_encoding=encoding_pb2.CompressedVideoEncoding(
                codec="h264",
                encoder="libx264",
                fps=round(fps),
                options=[
                    encoding_pb2.CompressedVideoEncoding.EncoderOption(name="crf", value="19"),
                    encoding_pb2.CompressedVideoEncoding.EncoderOption(name="gop_size", value="4"),
                    encoding_pb2.CompressedVideoEncoding.EncoderOption(name="bf", value="0"),
                ],
            ),
        )
    )


def double_list(values: np.ndarray) -> frame_pb2.DoubleList:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    msg = frame_pb2.DoubleList()
    msg.values.extend(values.tolist())
    return msg


def parse_double_list(msg: frame_pb2.DoubleList, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(msg.values, dtype=np.float64).reshape(shape)


def parse_frame_topic(topic: str) -> FieldKey:
    if not topic.startswith(PI_STREAM_FRAME_PREFIX):
        raise ValueError(f"Expected PiStream frame topic, got {topic!r}")
    publisher_id, key = topic[len(PI_STREAM_FRAME_PREFIX) :].split("/", 1)
    return FieldKey(publisher_id=publisher_id, key=key)


class ShardedMcapWriter:
    def __init__(
        self,
        output_dir: Path,
        field_specs: Iterable[FieldSpec],
    ) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._field_specs = {field.field_key: field for field in field_specs}
        self._seen_fields: set[FieldKey] = set()
        self._sequence_by_publisher: dict[str, int] = {}
        self._files: dict[int, Any] = {}
        self._writers: dict[int, mcap_protobuf.writer.Writer] = {}
        self._closed = False

        for shard in sorted({field.shard for field in self._field_specs.values()}):
            output_path = self._output_dir / f"episode_part{shard}.mcap"
            file = output_path.open("wb")
            writer = mcap_protobuf.writer.Writer(file, chunk_size=MCAP_CHUNK_SIZE)
            self._files[shard] = file
            self._writers[shard] = writer

    def __enter__(self) -> ShardedMcapWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def next_sequence(self, publisher_id: str) -> int:
        sequence = self._sequence_by_publisher.get(publisher_id, 0) + 1
        self._sequence_by_publisher[publisher_id] = sequence
        return sequence

    def write(
        self,
        field: FieldKey,
        message: protobuf_message.Message,
        *,
        timestamp_ns: int,
        sequence_id: int,
    ) -> None:
        if self._closed:
            raise ValueError("MCAP writer is closed.")
        spec = self._field_specs[field]
        if not isinstance(message, spec.message_type):
            raise TypeError(f"Expected {spec.message_type.__name__} for {field}, got {type(message).__name__}")
        self._writers[spec.shard].write_message(
            spec.channel_topic,
            message=message,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            sequence=sequence_id,
        )
        self._seen_fields.add(field)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for shard, writer in self._writers.items():
            try:
                descriptor = descriptor_pb2.PiStreamDescriptor(
                    fields=[
                        spec.to_descriptor_proto()
                        for spec in self._field_specs.values()
                        if spec.shard == shard and spec.field_key in self._seen_fields
                    ]
                )
                # Match Monopi's native raw-log saver: stream descriptors are metadata records
                # published at timestamp 0 in every shard.
                writer.write_message(
                    PI_STREAM_DESCRIPTOR_TOPIC,
                    message=descriptor,
                    log_time=0,
                    publish_time=0,
                    sequence=0,
                )
                writer.finish()
            except BaseException as exc:
                errors.append(exc)
        for file in self._files.values():
            try:
                file.close()
            except BaseException as exc:
                errors.append(exc)
        self._closed = True
        if errors:
            raise ExceptionGroup("Failed to close MCAP writer", errors)


@dataclasses.dataclass(frozen=True)
class EncodedVideoFrame(Generic[_UserdataT]):
    packets: list[bytes]
    pts: int
    dts: int
    is_keyframe: bool
    iframe_offset: int
    extradata: bytes | None
    userdata: _UserdataT | None = None

    def to_proto(self) -> frame_pb2.VideoFrame:
        return frame_pb2.VideoFrame(
            packets=self.packets,
            dts=self.dts,
            pts=self.pts,
            iframe_offset=self.iframe_offset,
            extradata=self.extradata,
        )


class H264VideoEncoder(Generic[_UserdataT]):
    def __init__(self, *, width: int, height: int, fps: float) -> None:
        self._raw_data = _ClearableBytesIO()
        self._container = av.open(self._raw_data, format="h264", mode="w")
        self._stream = self._container.add_stream("libx264", rate=round(fps))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.gop_size = 4
        self._stream.options = {"crf": "19", "bf": "0", "tune": "zerolatency"}
        self._next_pts_in = 0
        self._userdata: dict[int, _UserdataT] = {}
        self._iframe_offset = 0
        self._finalized = False

    def __del__(self) -> None:
        if hasattr(self, "_finalized") and not self._finalized:
            self.finalize()

    def encode_rgb(
        self,
        image: np.ndarray,
        *,
        userdata: _UserdataT | None = None,
    ) -> list[EncodedVideoFrame[_UserdataT]]:
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 RGB image, got shape={image.shape}, dtype={image.dtype}")
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        packets = self._stream.encode(frame)
        for packet in packets:
            self._container.mux(packet)
        if userdata is not None:
            self._userdata[self._next_pts_in] = userdata
        self._next_pts_in += 1
        self._raw_data.clear_buffer()
        return self._group_packets(packets)

    def finalize(self) -> list[EncodedVideoFrame[_UserdataT]]:
        if self._finalized:
            return []
        packets = self._stream.encode(None)
        for packet in packets:
            self._container.mux(packet)
        frames = self._group_packets(packets)
        self._container.close()
        self._finalized = True
        return frames

    def _group_packets(self, packets: list[av.packet.Packet]) -> list[EncodedVideoFrame[_UserdataT]]:
        frames: list[EncodedVideoFrame[_UserdataT]] = []
        for packet in packets:
            if packet.pts is None:
                raise RuntimeError("Expected encoded video packet with non-None pts.")
            if packet.dts is None:
                raise RuntimeError("Expected encoded video packet with non-None dts.")
            if not frames or frames[-1].dts != packet.dts:
                iframe_offset = 0 if packet.is_keyframe else self._iframe_offset + 1
                frames.append(
                    EncodedVideoFrame(
                        packets=[bytes(packet)],
                        pts=packet.pts,
                        dts=packet.dts,
                        is_keyframe=packet.is_keyframe,
                        iframe_offset=iframe_offset,
                        extradata=self._stream.codec_context.extradata,
                        userdata=self._userdata.pop(packet.pts, None),
                    )
                )
                self._iframe_offset = iframe_offset
            else:
                frames[-1].packets.append(bytes(packet))
        return frames


class H264VideoDecoder:
    def __init__(self) -> None:
        logging.getLogger("libav").setLevel(logging.CRITICAL)
        self._raw_data = io.BytesIO()
        self._container = av.open(self._raw_data, format="h264", mode="r")
        self._stream = self._container.streams.video[0]
        self._codec = self._stream.codec_context
        self._cur_pos = 0
        self._prev_dts: int | None = None
        self._pending_frames: deque[tuple[bytes, bytes | None, int]] = deque()
        self._finalized = False

    def __del__(self) -> None:
        if hasattr(self, "_finalized") and not self._finalized:
            self.finalize()

    def decode_to_rgb(self, video_frame: frame_pb2.VideoFrame) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        extradata = video_frame.extradata or None
        for packet in video_frame.packets:
            images.extend(frame.to_ndarray(format="rgb24") for frame in self.decode(packet, extradata, video_frame.dts))
        return images

    def decode(self, data: bytes, extradata: bytes | None, dts: int | None = None) -> list[av.VideoFrame]:
        if dts is not None and self._prev_dts is not None and dts != self._prev_dts + 1:
            self._pending_frames.append((data, extradata, dts))
            return []

        self._raw_data.write(data)
        self._raw_data.seek(self._cur_pos)
        if extradata is not None:
            self._codec.extradata = extradata

        frames: list[av.VideoFrame] = []
        for packet in self._container.demux():
            if packet.size == 0:
                continue
            self._cur_pos += packet.size
            for frame in self._stream.decode(packet):
                if isinstance(frame, av.VideoFrame):
                    frames.append(frame)
                else:
                    logging.warning("Unsupported decoded frame type: %s", type(frame))

        remaining = self._raw_data.getvalue()[self._cur_pos :]
        self._raw_data.seek(0)
        self._raw_data.write(remaining)
        self._raw_data.truncate()
        self._cur_pos = 0
        self._prev_dts = dts

        if self._pending_frames:
            if len(self._pending_frames) > 10:
                raise RuntimeError("Too many pending video frames.")
            pending = list(self._pending_frames)
            self._pending_frames.clear()
            for pending_data, pending_extradata, pending_dts in pending:
                frames.extend(self.decode(pending_data, pending_extradata, pending_dts))
        return frames

    def finalize(self) -> list[av.VideoFrame]:
        if self._finalized:
            return []
        frames = [frame for frame in self._stream.decode(None) if isinstance(frame, av.VideoFrame)]
        self._container.close()
        self._finalized = True
        return frames


def iter_mcap_messages(episode_dir: Path):
    for path in sorted(episode_dir.glob("episode_part*.mcap")):
        with path.open("rb") as file:
            reader = make_reader(file)
            yield from reader.iter_messages(log_time_order=False)


class _ClearableBytesIO(io.BytesIO):
    def clear_buffer(self) -> None:
        self.close()
        super().__init__(b"")
