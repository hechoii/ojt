import os

dirs = [
    "src/models", "src/data", "src/utils",
    "tests", "notebooks", "configs",
    "data/raw", "data/processed",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    init = os.path.join(d, "__init__.py")
    if not os.path.exists(init) and not d.startswith("data") and not d.startswith("notebooks") and not d.startswith("configs"):
        open(init, "w").close()

print("✅ Project structure created!")