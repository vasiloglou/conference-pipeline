from conference_pipeline.audio_chunker import size_safe_chunk_duration_seconds


def test_size_safe_chunk_duration_caps_high_bitrate_recordings() -> None:
    # A 100 MiB recording lasting 1,000 seconds averages 104,858 bytes/s;
    # a 16 MiB target therefore permits at most 160-second chunks.
    assert (
        size_safe_chunk_duration_seconds(
            source_byte_count=100 * 1024 * 1024,
            duration_seconds=1000.0,
            requested_chunk_duration=900,
        )
        == 160
    )


def test_size_safe_chunk_duration_retains_requested_duration_without_metadata() -> None:
    assert (
        size_safe_chunk_duration_seconds(
            source_byte_count=100,
            duration_seconds=None,
            requested_chunk_duration=900,
        )
        == 900
    )
