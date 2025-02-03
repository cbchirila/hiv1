
import os

class Obj:
    def create_directory(self,directory_path):
        if os.path.exists(directory_path):
            return None
        else:
            try:
                os.makedirs(directory_path)
            except:
                return None
            return directory_path
