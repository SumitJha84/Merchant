"""
patch_app.py - patches screen elif conditions to use emoji names matching the sidebar
"""
with open("dashboard/app.py", encoding="utf-8", errors="replace") as f:
    content = f.read()

replacements = [
    ('elif screen == "Merchant Drilldown":', 'elif screen == "\U0001f50d Merchant Drilldown":'),
    ('elif screen == "Watchlist":', 'elif screen == "\U0001f6a8 Watchlist":'),
    ('elif screen == "What-If Simulator":', 'elif screen == "\u2699\ufe0f What-If Simulator":'),
    ('elif screen == "Cluster Explorer":', 'elif screen == "\U0001f5fa\ufe0f Cluster Explorer":'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"Patched: {old[:40]}")
    else:
        print(f"NOT FOUND: {old[:40]}")

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done. Lines:", len(content.split("\n")))
