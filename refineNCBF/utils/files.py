import os

refineNCBF_dir = os.path.join(os.path.abspath(__file__).split('/refineNCBF/')[0], 'refineNCBF')
nbkm_dir = os.path.join(os.path.abspath(__file__).split('/dev/')[0], 'dev/local_reach/Neural-Barrier-Kinematic-Model')
visuals_data_directory = os.path.join(refineNCBF_dir, 'data', 'visuals')

FilePath = str
FilePathRelative = FilePath
FilePathAbsolute = FilePath


def construct_refine_ncbf_path(relative_path: FilePathRelative) -> FilePathAbsolute:
    return os.path.join(refineNCBF_dir, relative_path)


def construct_nbkm_path(relative_path: FilePathRelative) -> FilePathAbsolute:
    return os.path.join(nbkm_dir, relative_path)


def check_if_file_exists(file_path: FilePathRelative) -> bool:
    return os.path.isfile(construct_refine_ncbf_path(file_path))


def generate_unique_filename(header: str, extension: str) -> str:
    import datetime
    now = datetime.datetime.now()
    return f'{header}_{now.strftime("%Y%m%d_%H%M%S")}.{extension}'


def remove_file(file_path: FilePathRelative) -> None:
    os.remove(construct_refine_ncbf_path(relative_path=file_path))
