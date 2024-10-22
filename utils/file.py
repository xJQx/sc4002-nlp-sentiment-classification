from pathlib import Path

import dill


def save_to_local_file(file_path: Path, obj: any):
    print("Saving object to local...")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        dill.dump(obj, f)

    print("Object saved to local!")


def load_from_local_file(file_path: Path) -> any:
    print("Loading object from local...")

    with file_path.open("rb") as f:
        obj = dill.load(f)

    print("Object loaded from local!")
    return obj
