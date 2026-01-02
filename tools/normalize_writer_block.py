from pathlib import Path


def main() -> None:
    """Normalise the FFMpegWriter block in make_3d_animation_20190127.

    This helper rewrites the small block that configures Matplotlib's
    FFMpegWriter so that it has clean, space-only indentation and the desired
    bitrate/ffmpeg arguments, independent of whatever ad-hoc edits we tried
    while tuning.
    """

    path = Path("examples/make_3d_animation_20190127.py")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    target_block = [
        "    writer = FFMpegWriter(",
        "        fps=15,",
        "        bitrate=2000,  # target ~2 Mbit/s so the 40 s animation is ~8–12 MB on disk",
        "        extra_args=['-minrate', '2000k', '-maxrate', '2000k', '-bufsize', '4000k'],",
        "    )",
    ]

    for i, line in enumerate(lines):
        if "writer = FFMpegWriter(" in line:
            if i + 4 >= len(lines):
                raise RuntimeError("Unexpected file structure around FFMpegWriter block")
            lines[i : i + 5] = target_block
            print(f"Normalised FFMpegWriter block starting at line {i + 1}")
            break
    else:
        raise RuntimeError("Could not find FFMpegWriter block to normalise")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

