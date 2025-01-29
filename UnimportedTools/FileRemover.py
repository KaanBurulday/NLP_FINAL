import os
import pathlib


# Created for creating a smaller data sample.

assignment_data_path = f"{pathlib.Path().resolve()}\\data_sample"

def delete_data():
    base_path = pathlib.Path(assignment_data_path)
    for folder_path in base_path.iterdir():
        count = 0
        if folder_path.is_dir():
            for file_path in folder_path.iterdir():
                if count == 40:
                    break
                count += 1
                if file_path.is_file():
                    os.remove(file_path)


delete_data()