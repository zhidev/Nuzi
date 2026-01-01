import json
import os



def save_data(data, filename):
    #Writes dictionary data to file name
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def load_data(filename):
    if not os.path.exists(filename) or os.stat(filename).st_size==0:
        return {} # Return empty dict if file is empty or doesnst exist
    with open(filename, 'r') as f:
        return json.load(f)

def append_data(split_name,zone_name,trainer_name,input_data):
    filename = f"Split_Frags\\{split_name}"
    print(filename)
    data = load_data(filename)

    if zone_name not in data:
        data[zone_name] = {}

    print(f"Data is {input_data}")
    data[zone_name][trainer_name] = input_data
    print(f"Updated data: {data}")
    save_data(data, filename)

# def save_fight_data(split_name,zone_name,trainer_name, 
#                     pokemon1, pokemon2, pokemon3, 
#                     pokemon4, pokemon5, pokemon6, comment):
def save_fight_data(split_name, zone_name, trainer_name, fight_data):
    # fight_data ={
    #     'pokemon1' : pokemon1,
    #     'pokemon2' : pokemon2,
    #     'pokemon3' : pokemon3,
    #     'pokemon4' : pokemon4,
    #     'pokemon5' : pokemon5,
    #     'pokemon6' : pokemon6,
    #     'comment' : comment
    # }

    # party_string = party_string.strip()
    # print(repr(party_string))

    # regex_pattern = re.compile(r"\[(?P<pokemon1>.+)\] \[(?P<pokemon2>.+)\] \[(?P<pokemon3>.+)\] \[(?P<pokemon4>.+)\] \[(?P<pokemon5>.+)\] \[(?P<pokemon6>.+)\]")
    # match = re.search(regex_pattern, party_string)
    # print(f"Match is {match.groups()}")
    # comment_string = comment_string.strip()
    # if match: 
    #     print(match.group(1))
    #     print(match.group(2))
    #     print(match.group(3))
    #     print(match.group(4))
    #     print(match.group(5))
    #     print(match.group(6))
    #     fc.save_fight_data(split_name, zone_name, trainer_name, match.group(1), 
    #                        match.group(2), match.group(3), match.group(4), match.group(5),
    #                        match.group(6), comment_string)
    # else:
    #     print("Invalid match")
    #     print(repr(party_string))

    print("Properly in save fight data of data logic")
    print(f"{split_name}, {zone_name}, {trainer_name}, {fight_data}")


    print(f"In frag control save_fight_data. Data is {fight_data}")
    append_data(split_name,zone_name,trainer_name, fight_data)
    # }
    



if __name__ == "__main__":
    # fight_data = {
    #     'comment' : 'blep',
    #     'pokemon1' : 'Mareep'
    # }
    # trainer_data ={
    #     'Billy3' : fight_data
    # }
    # #append_data("Brawly_Split", "Route 102", '', fight_data)
    # save_fight_data("Brawly_Split", "Route 102", "Joe Poata", "As", "Bw", "C",
    #                 "Ds", "aE", "Fw", "cakes")
    pass