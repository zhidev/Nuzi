import json
from jsonparse import find_key, find_keys, find_key_chain, find_key_value, find_value

file_path = "Split_json\Brawly_Split\Route_106.json"

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from the file '{file_path}'. Check if the file has a valid JSON format.")

#json search
split_path = ""
def set_split(split):
    split_path = f"Split_json\\{split}\\"

def get_trainer_list_in_zone(data, zone):
    file_path = split_path.append(f"{zone}.json")
                                  


trainer_list = find_key(data, "trainer")

print(trainer_list)