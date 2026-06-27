import os

src = open("dashboard/app_raw.txt", encoding="utf-8", errors="replace").read()
lines = src.split("\n")[:222]
with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Restored", len(lines), "lines")
