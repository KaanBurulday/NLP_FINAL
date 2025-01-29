import cProfile
import pathlib
import pstats

output_folder_path = f"{pathlib.Path().resolve()}\\profile_outputs"
def view_outputs():
    base_path = pathlib.Path(output_folder_path)
    for file_path in base_path.iterdir():
        print(file_path.name)
        p = pstats.Stats(f"{output_folder_path}\\{file_path.name}")
        p.sort_stats(pstats.SortKey.TIME).print_stats(10)
        print("--------------------------------------------------------------------------------------------------")

view_outputs()