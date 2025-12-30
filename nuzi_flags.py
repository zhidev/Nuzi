move_set = [
    "Counter", "Mirror Coat", "Mirror Burst", "Beak Blast", "Destiny Bond",
    "Pursuit", "Protect", "Icy Wind", "Electroweb", "Bulldoze", "Low Sweep",
    "Rock Tomb"

]

item_set = [
    "Focus Sash", "Focus Band", "Custap Berry", "Weakness Policy", "Eject Button",
    "Eject Pack", 
]

ability_set = [
    "Sheer Force", "Shield Dust", "Inner Focus", "Intimidate", "Unnerve", "Sturdy",
    "Steam Engine", "Disguise", "Prankster"
]

def return_ability_flags(ability):
    abilities_flagged = set(ability) & set(ability_set)
    return_string = f"The following abilities have been flagged: {" ".join(abilities_flagged)}\n"
    return return_string

def return_moves_flags(moves):
    moves_flagged = set(moves) & set(move_set)
    return_string = f"The following moves have been flagged: {" ".join(moves_flagged)}\n"
    return return_string

def return_items_flags(items):
    items_flagged = set(items) & set(item_set)
    return_string = f"The following items have been flagged: {" ".join(items_flagged)}\n    "
    return return_string

def return_string_of_flags_with_ability_moves_items(ability, moves, items):
    return_string = return_ability_flags(ability)
    return_string += return_moves_flags(moves)
    return_string += return_items_flags(items)
    return return_string




if __name__ == "__main__":
    # sample_ability = ["Intimidate", "fake", "potato", "pie", "Sturdy"]
    # sample_items = ["potato", "Focus Sash", "a", "Custap Berry"]
    # sample_moves = ["Dodge", "Counter"]
    
    # print()

    pass