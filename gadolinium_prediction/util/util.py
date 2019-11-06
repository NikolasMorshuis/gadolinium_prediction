import os
import sys

name = sys.executable
name = name.split('/')[2]

class Dataoperations():
    def __init__(self):
        pass

    def walklevel(self, some_dir, level=1):
        """
        Walks subdirectories level steps down and returns paths to all objects at this level
        Credits to Jona
        """
        some_dir = some_dir.rstrip(os.path.sep)
        assert os.path.isdir(some_dir)
        num_sep = some_dir.count(os.path.sep)
        for root, dirs, files in os.walk(some_dir):
            num_sep_this = root.count(os.path.sep)
            if num_sep_this == level:
                yield root, dirs, files
            if num_sep + level <= num_sep_this:
                del dirs[:]

    def get_subdirs(self, dir, level):
        """
        Returns list of subdirs at level
        Credits to Jona
        """
        subdirs = []
        for root, dirs, files in self.walklevel(dir, level=level):
            subdirs.append(root)
        # print(subdirs)
        return subdirs

    def get_filenames(self, dir, level):
        filenames = []
        for root, dirs, files in self.walklevel(dir, level=level):
            filenames.append(files)
        return(filenames)

    def get_directories(self, dir, level):
        filenames = []
        for root, dirs, files in self.walklevel(dir, level=level):
            filenames.append(dirs)
        return(filenames)

    def get_patientnumbers(self):
        dirs = self.get_subdirs('/home/'+name+'/Documents/essen_data', level=5)
        patients = []
        for dir in dirs:
            patient = dir.split('/')[-1]
            try:
                float(patient)
                patients.append(patient)
            except ValueError:
                print(patient)
        return(patients)

    def generate_list_of_files(self, path, level):
        test = self.walklevel(path, level)
        return(test)
