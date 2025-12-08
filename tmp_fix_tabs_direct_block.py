from pathlib import Path

TARGET = Path("examples/diagnostic_sav_vs_mmsmp_20190127.py")
lines = TARGET.read_text(encoding="utf-8").splitlines()

for i in range(283, 292):  # 1-based line numbers
    idx = i - 1
    line = lines[idx]
    if line.startswith("\t"):
        lines[idx] = line[1:]

TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Removed leading tab from lines 283-291 in", TARGET)

