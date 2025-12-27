import ttkbootstrap as tb
from ttkwidgets.autocomplete import AutocompleteCombobox

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
root.geometry("800x600")


# Create Functions for button
def split_selected():
    my_label.config(text=f"You clicked on {split_combobox.get()}!")


# Create Binding Functions
def split_combobox_click_bind(e):
    split_name = split_combobox.get()
    zone_combobox.config(values=trainer_data[split_name])
    zone_combobox.set(f"Select a Zone from the {split_name}")


def zone_combobox_click_bind(e):
    zone_name = zone_combobox.get()
    trainer_combobox.config(values=encounter_data[zone_combobox.get()])
    trainer_combobox.set(f"Select the trainer from the Zone: {zone_name}")
    trainer_combobox.event_generate("<<ComboboxSelected>>") 

def trainer_combobox_click_bind(e):
    my_label.config(text=f"{trainer_combobox.get()}")

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
    root, bootstyle="success", values=trainer_data["split_names"]
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


# def split_menu_action(x):
#     #my_label.config(text=f"{trainer_data[x]}")
#     my_label.config(text =f"{x}")
#     for zone_items in  trainer_data[x]:
#         zone_inside_menu.add_radiobutton(label=zone_items, value=zone_items,
#                                          command=lambda x=zone_items: zone_menu_action(x))
#     zone_menu['menu'] = zone_inside_menu

# def zone_menu_action(x):
#     my_label.config(text=f"{x}")

# #split menu
# split_menu = tb.Menubutton(root, bootstyle="primary", text="Pick Current Split")
# split_menu.pack(pady=40)

# zone_menu = tb.Menubutton(root, bootstyle="secondary", text="Pick Current Zone")
# zone_menu.pack(pady=60)
# #Create basic menu
# split_inside_menu = tb.Menu(split_menu)
# zone_inside_menu = tb.Menu(zone_menu)

# #Add items to our Menu
# for split_items in trainer_data["split_names"]:
#     split_inside_menu.add_radiobutton(label=split_items,value=split_items,
#                                        command=lambda x=split_items: split_menu_action(x))

# #Associate the inside menu with the menu button
# split_menu['menu'] = split_inside_menu

root.mainloop()
