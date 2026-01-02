from pathlib import Path
import re


def main() -> None:
    path = Path("examples/make_3d_animation_20190127.py")
    text = path.read_text(encoding="utf-8")

    # Normalise indentation for the FFMpegWriter block so it uses spaces only.
    text = re.sub(r"\n\s*writer = FFMpegWriter\(", "\n    writer = FFMpegWriter(", text)
    text = re.sub(r"\n\s*fps=15,", "\n        fps=15,", text)
    text = re.sub(r"\n\s*bitrate=2000,", "\n        bitrate=2000,", text)
    text = re.sub(
        r"\n\s*extra_args=\[\"-minrate\", \"2000k\", \"-maxrate\", \"2000k\", \"-bufsize\", \"4000k\"],",
        "\n        extra_args=[\"-minrate\", \"2000k\", \"-maxrate\", \"2000k\", \"-bufsize\", \"4000k\"],",
        text,
    )
    # The closing parenthesis of the FFMpegWriter call should be indented one
    # level (4 spaces) from the function body.
    text = re.sub(r"\n\s*\)", "\n    )", text, count=1)

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

