import json
import os

COUNSELORS_PATH = os.path.join(os.path.dirname(__file__), "../resources/counselors.json")

with open(COUNSELORS_PATH, "r") as f:
    COUNSELORS = json.load(f)

def find_counselor_by_program(program_name: str):
    program_name = program_name.strip().lower()
    for entry in COUNSELORS:
        for prog in entry["programs"]:
            if prog.lower() == program_name:
                return entry["counselor"]
    return None

if __name__ == "__main__":
    print(find_counselor_by_program("MSc Data Analytics"))

