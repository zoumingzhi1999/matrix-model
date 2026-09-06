"""Read raw side-scan or multichannel echosounder XTF into an intensity image."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyxtf
from pyxtf.xtf_ctypes import (
    XTFFileHeader,
    XTFPacketStart,
    XTFPingChanHeader,
    XTFPingHeader,
    sample_format_dtype,
    xtf_dtype,
)


def _channel_type(file_header, channel_number: int):
    value = file_header.ChanInfo[channel_number].TypeOfChannel
    try:
        return pyxtf.XTFChannelType(int(value))
    except ValueError:
        return value


def _dense_channel(
    rows: list[np.ndarray],
    weights: list[int],
    channel_type,
) -> np.ndarray:
    maximum = max(len(row) for row in rows)
    output = np.zeros((len(rows), maximum), dtype="float32")
    for row_index, (row, weight) in enumerate(zip(rows, weights)):
        weighted = row.astype("float32", copy=False) * float(2.0 ** -weight)
        if channel_type == pyxtf.XTFChannelType.port:
            start = maximum - len(row)
        elif channel_type == pyxtf.XTFChannelType.stbd:
            start = 0
        else:
            start = (maximum - len(row)) // 2
        output[row_index, start:start + len(row)] = weighted
    if channel_type == pyxtf.XTFChannelType.port:
        output = np.fliplr(output)
    return output


def _read_generic_sonar_packets(xtf_path: Path):
    """Read non-port/starboard XTF sonar packets using pyxtf structures."""
    ping_rows = []
    file_size = xtf_path.stat().st_size
    with xtf_path.open("rb") as handle:
        file_header = XTFFileHeader.create_from_buffer(handle)
        while handle.peek(1):
            packet_start = handle.tell()
            start = XTFPacketStart()
            if handle.readinto(start) < ctypes.sizeof(XTFPacketStart):
                raise RuntimeError("XTF file ended inside a packet header.")
            record_bytes = int(start.NumBytesThisRecord)
            if record_bytes < ctypes.sizeof(XTFPacketStart):
                raise ValueError(f"Invalid XTF packet size at byte {packet_start}.")
            if packet_start + record_bytes > file_size:
                raise ValueError(
                    f"XTF packet at byte {packet_start} extends past end of file."
                )
            if int(start.HeaderType) == int(pyxtf.XTFHeaderType.sonar):
                handle.seek(packet_start)
                ping = XTFPingHeader()
                if handle.readinto(ping) < ctypes.sizeof(XTFPingHeader):
                    raise RuntimeError("XTF file ended inside a sonar ping header.")
                remaining = record_bytes - ctypes.sizeof(XTFPingHeader)
                channels = []
                for _ in range(int(ping.NumChansToFollow)):
                    channel_header = XTFPingChanHeader()
                    if handle.readinto(channel_header) < ctypes.sizeof(XTFPingChanHeader):
                        raise RuntimeError("XTF file ended inside a channel header.")
                    remaining -= ctypes.sizeof(XTFPingChanHeader)
                    channel_number = int(channel_header.ChannelNumber)
                    if not 0 <= channel_number < len(file_header.ChanInfo):
                        raise ValueError(
                            f"XTF channel number {channel_number} is out of range."
                        )
                    channel_info = file_header.ChanInfo[channel_number]
                    sample_count = int(channel_header.NumSamples)
                    if sample_count <= 0:
                        sample_count = int(channel_info.Reserved)
                    bytes_per_sample = int(channel_info.BytesPerSample)
                    sample_bytes = sample_count * bytes_per_sample
                    if sample_bytes > remaining:
                        raise ValueError(
                            "XTF channel samples exceed the current packet size."
                        )
                    dtype = sample_format_dtype.get(
                        int(channel_info.SampleFormat),
                        xtf_dtype.get(bytes_per_sample),
                    )
                    if dtype is None:
                        raise ValueError(
                            f"Unsupported XTF sample width: {bytes_per_sample}."
                        )
                    raw = handle.read(sample_bytes)
                    if len(raw) != sample_bytes:
                        raise RuntimeError("XTF file ended inside channel samples.")
                    remaining -= sample_bytes
                    channels.append({
                        "channel_number": channel_number,
                        "weight": int(channel_header.Weight),
                        "samples": np.frombuffer(raw, dtype=dtype).copy(),
                    })
                ping_rows.append(channels)
            handle.seek(packet_start + record_bytes)

    if not ping_rows:
        raise ValueError(f"XTF file contains no sonar ping packets: {xtf_path}")
    channel_counts = {len(ping) for ping in ping_rows}
    if len(channel_counts) != 1:
        raise ValueError(
            "XTF sonar ping channel counts are inconsistent; refusing to "
            "discard channels."
        )
    channel_count = channel_counts.pop()
    if channel_count < 1:
        raise ValueError(f"XTF sonar pings contain no sample channels: {xtf_path}")
    images = []
    metadata = []
    for channel_index in range(channel_count):
        channel_number = int(ping_rows[0][channel_index]["channel_number"])
        if any(
            int(ping[channel_index]["channel_number"]) != channel_number
            for ping in ping_rows
        ):
            raise ValueError(
                "XTF sonar channel order changes between pings; refusing to "
                "combine different channels."
            )
        channel_type = _channel_type(file_header, channel_number)
        rows = [ping[channel_index]["samples"] for ping in ping_rows]
        weights = [int(ping[channel_index]["weight"]) for ping in ping_rows]
        image = _dense_channel(rows, weights, channel_type)
        images.append(image)
        metadata.append({
            "channel_index": channel_index,
            "channel_number": channel_number,
            "channel_type": getattr(channel_type, "name", str(channel_type)),
            "samples_per_ping_min": min(len(row) for row in rows),
            "samples_per_ping_max": max(len(row) for row in rows),
        })
    return file_header, images, metadata, len(ping_rows)


def load_xtf_sonar_image(path: str | Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Decode all XTF sonar pings without resizing or discarding ping rows."""
    xtf_path = Path(path)
    if xtf_path.suffix.lower() != ".xtf":
        raise ValueError("Sonar input must be a raw .xtf file.")
    if not xtf_path.is_file():
        raise FileNotFoundError(f"XTF sonar file does not exist: {xtf_path}")

    decoder_mode = "pyxtf_high_level"
    try:
        file_header, packets = pyxtf.xtf_read(
            str(xtf_path), types=[pyxtf.XTFHeaderType.sonar]
        )
        pings = packets.get(pyxtf.XTFHeaderType.sonar, [])
        if not pings:
            raise ValueError(f"XTF file contains no sonar ping packets: {xtf_path}")
        channel_counts = {len(ping.data) for ping in pings}
        if len(channel_counts) != 1:
            raise ValueError(
                "XTF sonar ping channel counts are inconsistent; refusing to "
                "discard channels."
            )
        channel_count = channel_counts.pop()
        if channel_count < 1:
            raise ValueError(f"XTF sonar pings contain no sample channels: {xtf_path}")

        channel_images = []
        channel_metadata = []
        for channel_index in range(channel_count):
            image = pyxtf.concatenate_channel(
                pings,
                file_header=file_header,
                channel=channel_index,
                weighted=True,
            ).astype("float32", copy=False)
            channel_number = int(
                pings[0].ping_chan_headers[channel_index].ChannelNumber
            )
            channel_type = _channel_type(file_header, channel_number)
            if channel_type == pyxtf.XTFChannelType.port:
                image = np.fliplr(image)
            channel_images.append(image)
            channel_metadata.append({
                "channel_index": channel_index,
                "channel_number": channel_number,
                "channel_type": getattr(channel_type, "name", str(channel_type)),
                "samples_per_ping_min": int(image.shape[1]),
                "samples_per_ping_max": int(image.shape[1]),
            })
        ping_count = len(pings)
    except IndexError:
        decoder_mode = "pyxtf_structured_multichannel_fallback"
        file_header, channel_images, channel_metadata, ping_count = (
            _read_generic_sonar_packets(xtf_path)
        )
        channel_count = len(channel_images)

    # Port is mirrored away from nadir and starboard extends in the opposite
    # direction. Concatenation preserves every ping and every range sample.
    raw_image = np.concatenate(channel_images, axis=1)
    finite = np.isfinite(raw_image)
    if not finite.any():
        raise ValueError(f"XTF sonar image has no finite intensities: {xtf_path}")
    raw_image = np.where(finite, np.maximum(raw_image, 0.0), 0.0)
    log_image = np.log1p(raw_image, dtype="float32")
    nonzero = log_image[log_image > 0.0]
    scale_values = nonzero if nonzero.size else log_image.reshape(-1)
    low, high = np.percentile(scale_values, [1.0, 99.0])
    if not np.isfinite(high) or high <= low:
        low = float(np.min(scale_values))
        high = float(np.max(scale_values))
    if high <= low:
        normalized = np.zeros_like(log_image, dtype="float32")
    else:
        normalized = np.clip((log_image - low) / (high - low), 0.0, 1.0)
        normalized = normalized.astype("float32", copy=False)

    metadata = {
        "source": xtf_path.name,
        "format": "XTF",
        "decoder": f"pyxtf {getattr(pyxtf, '__version__', '1.5.0')}",
        "decoder_mode": decoder_mode,
        "sonar_ping_count": ping_count,
        "channel_count": channel_count,
        "channels": channel_metadata,
        "output_height": int(normalized.shape[0]),
        "output_width": int(normalized.shape[1]),
        "resize_applied": False,
        "intensity_transform": "weighted samples -> log1p -> percentile 1-99 clipping",
        "intensity_percentile_low": float(low),
        "intensity_percentile_high": float(high),
    }
    return normalized, metadata
