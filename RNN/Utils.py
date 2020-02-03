from datetime import datetime, timedelta
from pathlib import Path


def korea_time(time_format="%Y/%m/%d %H:%M:%S"):
    kt = datetime.utcnow() + timedelta(hours=9)
    if time_format is not None:
        return kt.strftime(time_format)
    return kt


def check_valid_path(file_path):
    if not file_path.exists():
        raise FileNotFoundError("\n[ {} ] 경로가 존재하지 않습니다.\n".format(file_path))


def create_folder(folder_location):
    root_folder = Path.cwd() if folder_location is None else Path(folder_location).resolve()
    check_valid_path(root_folder.parent)

    if not root_folder.exists():
        root_folder.mkdir()

    result_folder = root_folder / korea_time("%Y%m%d__%H%M%S")
    log_folder = result_folder / "logs"
    ckpt_folder = result_folder / "ckpt"
    image_folder = result_folder / "images"

    for folder in [result_folder, log_folder, ckpt_folder, image_folder]:
        if not folder.exists():
            folder.mkdir()

    return log_folder, ckpt_folder, image_folder
