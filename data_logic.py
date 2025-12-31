import trainer_data as td
import nuzi_flags as nflags
import numpy as np
import image_process as nuzi_ocr


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

if __name__ == "__main__":
    print(check_flags_for_trainer("Pokemon Breeder Lydia"))
    pass
