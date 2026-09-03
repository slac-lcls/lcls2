"""CPU-side discovery of detector payload layout in bigdata dgrams.

This module does not implement a general XTC parser. It derives the small
layout and routing description needed by the GPU raw-assembly stage.
"""


def detect_dgram_layout(dgram_bytes):
    """Return ``(segment_stride, raw_data_offset)`` for a bigdata dgram.

    The input has already been identified as an uncompressed area-detector
    dgram. Only the relevant Container and Shapes extents are inspected here.
    """
    segment_stride = int.from_bytes(dgram_bytes[32:36], "little")
    shapes_extent = int.from_bytes(dgram_bytes[44:48], "little")
    raw_data_offset = 36 + shapes_extent + 12
    return segment_stride, raw_data_offset


def segment_ids_in_l1_order(dgram, det_name):
    """Return detector segment IDs in decoded L1 child-XTC order."""
    detector_data = getattr(dgram, det_name, None)
    if detector_data is None:
        return []
    return [int(segment_id) for segment_id in detector_data.keys()]


def build_stream_segment_map(stream_bd_files, det_name="jungfrau"):
    """Discover detector segment order for every detector-bearing stream.

    ``DgramManager`` performs normal CPU XTC decoding. This function opens the
    first relevant L1Accept in each stream and records the resulting physical
    segment order for GPU raw assembly.
    """
    from psana.dgrammanager import DgramManager
    from psana.psexp.transitionid import TransitionId

    stream_segment_map = {}
    for stream_id, bd_file in stream_bd_files.items():
        dm = None
        try:
            dm = DgramManager([str(bd_file)])

            carries_detector = any(
                hasattr(getattr(config, "software", None), det_name)
                for config in dm.configs
            )
            if not carries_detector:
                continue

            for dgrams in dm:
                dgram = dgrams[0] if dgrams else None
                if dgram is None or not TransitionId.isEvent(dgram.service()):
                    continue
                segment_ids = segment_ids_in_l1_order(dgram, det_name)
                if segment_ids:
                    stream_segment_map[int(stream_id)] = segment_ids
                    break

            if int(stream_id) not in stream_segment_map:
                import warnings

                warnings.warn(
                    f"build_stream_segment_map: stream {stream_id} Configure "
                    f"contains {det_name!r}, but no detector L1Accept was found"
                )
        except Exception as exc:
            import warnings

            warnings.warn(
                f"build_stream_segment_map: could not read stream {stream_id} "
                f"({bd_file}): {exc}"
            )
        finally:
            if dm is not None:
                dm.close()

    return stream_segment_map
