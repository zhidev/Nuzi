import trainer_data as td
import nuzi_flags as nflags
import numpy as np
import image_process as nuzi_ocr
import frag_controls as fc
import re

def get_splits():
    return td.get_splits()


def get_zones_in_split(split_name):
    return td.get_zones_in_split(split_name)


def get_trainers_in_split(split_name):
    return td.get_trainers_in_split(split_name)


def get_trainers_in_split_in_zone(split_name, zone_name):
    return td.get_trainers_in_split_in_zone(split_name, zone_name)


def get_pokemon_from_trainers_in_split_in_zone(split_name, zone_name):
    return td.get_pokemon_from_trainers_in_split_in_zone(split_name, zone_name)


def get_trainer_dictionary(trainer_name):
    return td.get_trainer_dictionary(trainer_name)


def get_pokemon_from_trainer_name(trainer_name):
    return td.get_pokemon_from_trainer_name(trainer_name)


def get_pokemon_moves_from_trainer_name(trainer_name):
    return td.get_pokemon_moves_from_trainer_name(trainer_name)


def get_pokemon_items_from_trainer_name(trainer_name):
    return td.get_pokemon_items_from_trainer_name(trainer_name)


def get_pokemon_ability_from_trainer_name(trainer_name):
    return td.get_pokemon_ability_from_trainer_name(trainer_name)

def check_flags_for_trainer(trainer_name):
    
    pokemon_abilities = get_pokemon_ability_from_trainer_name(trainer_name)
    pokemon_items = get_pokemon_items_from_trainer_name(trainer_name)
    #we need to flatten multidimensional array into one single list
    multiarray = get_pokemon_moves_from_trainer_name(trainer_name)
    pokemon_moves = [item for sublist in multiarray for item in sublist]
    return nflags.return_string_of_flags_with_ability_moves_items(ability=pokemon_abilities,
                                                                  moves=pokemon_moves, items=pokemon_items)


def convert_PIL_to_cv_img(img):
    return nuzi_ocr.convert_PIL_to_cv_img(pil_img=img)

def ocr_image(img):
    return nuzi_ocr.ocr_image(img=img,flag=0)

def ocr_image_for_names(img):
    #flag 0 for list of all possible names
    name_list = nuzi_ocr.ocr_image(img=img,flag=0)
    bracketed_list = [f'[{item}]' for item in name_list]
    return " ".join(bracketed_list)

def save_fight_data(split_name,zone_name,trainer_name, 
                    party_string,comment_string):
    # fc.save_fight_data(split_name,zone_name,trainer_name,
    #                    pokemon1, pokemon2, pokemon3, pokemon4,
    #                    pokemon5, pokemon6,comments)
    print("In data logic save fight data")
    print("Properly in save fight data of data logic")

    party_string = party_string.strip()
    print(repr(party_string))
    comment_string = comment_string.strip()

    regex_pattern = re.compile(r"\[(?P<pokemon1>.+)\] \[(?P<pokemon2>.*)\] \[(?P<pokemon3>.*)\] \[(?P<pokemon4>.*)\] \[(?P<pokemon5>.*)\] \[(?P<pokemon6>.*)\]")
    
    #Walrus method s we do search/match and check if match is none in one line
    if match := regex_pattern.match(party_string):
        comment_entry = {'comment' : comment_string}
        fight_data = match.groupdict() | comment_entry
        fc.save_fight_data(split_name, zone_name, trainer_name, fight_data)
    else:
        print("Issue matching 6 ")
    
    
    # match = re.search(regex_pattern, party_string)
    # print(f"Match is {match.groups()}")
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
        # print(repr(party_string))

    print("Properly in save fight data of data logic")
    print(f"{split_name}, {zone_name}, {trainer_name}, {party_string}, {comment_string}")
    pass


if __name__ == "__main__":
    # print(check_flags_for_trainer("Pokemon Breeder Lydia"))

    party_string = "[P1] [p2] [P3as] [P4] [P5ity] [P6ixty]"
    # party_string = "what the fuck"

    regex_pattern = re.compile(r"\[(?P<pokemon1>.+)\] \[(?P<pokemon2>.+)\] \[(?P<pokemon3>.+)\] \[(?P<pokemon4>.+)\] \[(?P<pokemon5>.+)\] \[(?P<pokemon6>.+)\]")
    # regex_pattern = re.compile(r"what")
    match = re.match(regex_pattern, party_string)
    print(f"Match is {match.groups()}")



    pass
