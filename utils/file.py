import os
import dill

def _makeDir_if_does_not_exist(dir_path: str) -> bool:
    """
        Create directory path in local if `dir` is not found.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        return True
    return False

def save_to_local_file(file_path: str, object: any):
    print("Saving object to local...")
    if file_path[0] == "/":
        file_path = file_path[1:]
    file_directory = os.path.dirname(file_path)

    _makeDir_if_does_not_exist(file_directory)
        
    f = open(file_path, 'wb')
    dill.dump(object, f)
    f.close()

    print("Object saved to local!")

def load_from_local_file(file_path: str) -> any:
    print("Loading object from local...")

    if file_path[0] == "/":
        file_path = file_path[1:]
    
    f = open(file_path, 'rb')
    object = dill.load(f)
    f.close()

    print("Object loaded from local!")
    return object