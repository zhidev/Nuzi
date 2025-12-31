import trainer_data
import image_process as nuzi_ocr



# print(trainer_data.get_pokemon_from_trainers_in_split_in_zone("Norman", "Petalburg Gym"))

print(trainer_data.get_pokemon_from_trainer_name("Ruin Maniac Georgie"))
print(trainer_data.get_pokemon_moves_from_trainer_name("Ruin Maniac Georgie"))
print(trainer_data.get_pokemon_ability_from_trainer_name("Ruin Maniac Georgie"))
print(trainer_data.get_pokemon_items_from_trainer_name("Ruin Maniac Georgie"))

nuzi_ocr.ocr_image()