import os

root_dir = os.path.join(os.path.abspath(__file__).split('/refineNCBF/')[0], 'refineNCBF')

visuals_data_directory = os.path.join(root_dir, 'data', 'visuals')

FilePath = str
FilePathRelative = FilePath
FilePathAbsolute = FilePath


def construct_full_path(relative_path: FilePathRelative) -> FilePathAbsolute:
    return os.path.join(root_dir, relative_path)


def check_if_file_exists(file_path: FilePathRelative) -> bool:
    return os.path.isfile(construct_full_path(file_path))


def generate_unique_filename(header: str, extension: str) -> str:
    import datetime
    now = datetime.datetime.now()
    return f'{header}_{now.strftime("%Y%m%d_%H%M%S")}.{extension}'


def remove_file(file_path: FilePathRelative) -> None:
    os.remove(construct_full_path(relative_path=file_path))
