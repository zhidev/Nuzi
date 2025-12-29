import ttkbootstrap as tb
import trainer_data as td

level_list = ["Lu83", "Lu30", "Lu35", "Lu33", "Lu93", "Luz"]
hp_value_list = ["209/205", "130/266", "156/293", "178/255", "199/302", "76/76"]
name_list = ["Shorki", "Prina", "Cerbi", "feta", "Heary", "Celi", "Cancel"]


# Drop down options
game_splits = ["Grunt", "Museum"]
game_zones = ["starter", "beach", "cave"]
grunt_split = ["forest", "beach"]
museum_split = ["111", "museum", "krab"]

trainer_data = {
    "split_names": ["Grunt Split", "Museum Split"],
    "Grunt Split": ["112", "113"],
    "Museum Split": ["Museum"],
}

encounter_data = {
    "112" : ["Trainer 1, Trainer 2"],
    "113" : ["Trainer 3, Trainer 4"],
    "Museum" : ["Double"]
}


# 4 and 0 looks similar

if name_list[-1] == "Cancel":
    name_list.pop()


root = tb.Window(themename="solar")

root.title("Title!")
root.geometry("1480x1000")


# Create Functions for button
def split_selected():
    my_label.config(text=f"You clicked on {split_combobox.get()}!")


# Create Binding Functions
def split_combobox_click_bind(e):
    split_name = split_combobox.get()

    zone_combobox.config(values=td.get_zones_in_split(split_name))
    zone_combobox.set(f"Select a Zone from the {split_name} split")


def zone_combobox_click_bind(e):
    zone_name = zone_combobox.get()
    trainer_combobox.config(
        values=td.get_trainers_in_split_in_zone(split_name=split_combobox.get(),
                                                zone_name=zone_name))
    trainer_combobox.set(f"Select the trainer from the Zone: {zone_name}")
    trainer_combobox.event_generate("<<ComboboxSelected>>") 

def trainer_combobox_click_bind(e):
    trainer_name = trainer_combobox.get()
    my_label.config(text=f"Pokemons are: {td.get_pokemon_from_trainer_name(trainer_name=trainer_name)}")

# Styles
my_tk_styles = tb.Style()
my_tk_styles.configure("primary.TButton", font=("Helvetica, 18"))

# Colors: Default, primary, secondary, success, info, warning, danger
# light dark
# Create Labels
my_label = tb.Label(text="Papaya", font=("Helvetica", 28), bootstyle="primary")
my_label.pack(pady=50)


# Create Buttons
# my_button = tb.Button(text="Clickie",
#                     bootstyle="success",
#                    style="primary.TButton",
#                   width=20)
# my_button.pack(pady=20)

# Set Combo Default

# Create Split Combobox
split_combobox = tb.Combobox(
    root, bootstyle="success", values=td.get_splits()
)
split_combobox.pack(pady=20)
split_combobox.set("Select a Split")


# binding the split combobox
split_combobox.bind("<<ComboboxSelected>>", split_combobox_click_bind)

# Create the trainer Combobox that displays the trainers from split
zone_combobox = tb.Combobox(root, bootstyle="success")
zone_combobox.pack(pady=40)


# bind the trainer combobox
zone_combobox.bind("<<ComboboxSelected>>", zone_combobox_click_bind)

# Trainer combobox after seeing what zone we are

trainer_combobox = tb.Combobox(root, bootstyle="success")
trainer_combobox.pack(pady=10)
#Bind the trainer combobox
trainer_combobox.bind("<<ComboboxSelected>>", trainer_combobox_click_bind)


root.mainloop()
