"""Synthetic reusable acquisition lane: organizer-channel talk harvest.

Recommended for: the CVPR-shaped fixture conference (download-corpus talk lane).
"""

RECOMMENDED_FOR = ("WOMBAT",)


def harvest_channel_talks(channels, *, transport):
    talks = []
    for channel in channels:
        for video in transport(channel):
            talks.append({"channel": channel, "video_id": video["id"], "title": video.get("title")})
    return talks
