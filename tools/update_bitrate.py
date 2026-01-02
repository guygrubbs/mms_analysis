from pathlib import Path


def main() -> None:
    """Utility script to tweak the FFMpegWriter bitrate arguments.

    This is only used during development to tune the requested bitrate so that
    ffprobe reports an effective bitrate close to the desired value.
    """

    path = Path("examples/make_3d_animation_20190127.py")
    text = path.read_text(encoding="utf-8")

	# Bump the nominal bitrate up and adjust the matching ffmpeg CBR args.
	# We deliberately ask for a very high nominal rate here because ffmpeg's
	# effective bitrate for this simple animation tends to come out well below
	# the requested value. Empirically, requesting ~32 Mbit/s yields an actual
	# bitrate closer to the desired ~2 Mbit/s.
	text = text.replace("bitrate=8000,", "bitrate=32000,")
	text = text.replace(
	    '"-minrate", "8000k", "-maxrate", "8000k", "-bufsize", "16000k"',
	    '"-minrate", "32000k", "-maxrate", "32000k", "-bufsize", "64000k"',
	)

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

