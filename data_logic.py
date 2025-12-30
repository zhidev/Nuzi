import trainer_data as td
import nuzi_flags as nflags
import numpy as np

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


if __name__ == "__main__":
    print(check_flags_for_trainer("Pokemon Breeder Lydia"))
    pass
