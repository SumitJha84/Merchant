with open("assemble_app.py", encoding="utf-8") as f:
    content = f.read()

old = '"revenue_at_risk","drift_composite","alert_severity","alert_text"]].copy()'
new = '"revenue_at_risk","drift_composite","alert_severity"]].copy()'
content = content.replace(old, new)

with open("assemble_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched:", old in open("assemble_app.py").read())
