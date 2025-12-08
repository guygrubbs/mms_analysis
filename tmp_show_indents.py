from pathlib import Path

path = Path("examples/diagnostic_sav_vs_mmsmp_20190127.py")
lines = path.read_text(encoding="utf-8").splitlines()

for i in range(276, 292):
    line = lines[i - 1]
    prefix = line[:40]
    print(f"{i:3d}: {prefix!r}")

