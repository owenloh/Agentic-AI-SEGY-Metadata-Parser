from pathlib import Path

DEFAULT_SEGYS = Path('segypaths.txt')

def import_segys(text_file_path: Path):
    with open(text_file_path, 'r') as textfile:
        lines = textfile.readlines()
    
    # Strip newline characters from each line
    list_of_strings = [line.strip() for line in lines]
    
    # print(list_of_strings)
    return list_of_strings

