from pathlib import Path

TARGET = Path("examples/diagnostic_sav_vs_mmsmp_20190127.py")

text = TARGET.read_text(encoding="utf-8")
lines = text.splitlines()
out_lines = []

for lineno, line in enumerate(lines, start=1):
    # Remove the dangling legacy B_LMN snippet that lives between
    # lines 256–345 in the current version of the diagnostic script.
    if 256 <= lineno <= 345:
        continue
    out_lines.append(line)

TARGET.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print("Removed legacy B_LMN snippet from lines 256–345 in", TARGET)

