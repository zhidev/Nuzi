from tkinter import *  # noqa: F403

import ttkbootstrap as tb
from ttkbootstrap.scrolled import ScrolledText

import data_logic as td

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
    "112": ["Trainer 1, Trainer 2"],
    "113": ["Trainer 3, Trainer 4"],
    "Museum": ["Double"],
}


# 4 and 0 looks similar

if name_list[-1] == "Cancel":
    name_list.pop()


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
        values=td.get_trainers_in_split_in_zone(
            split_name=split_combobox.get(), zone_name=zone_name
        )
    )
    trainer_combobox.set(f"Select the trainer from the Zone: {zone_name}")
    # trainer_combobox.event_generate("<<ComboboxSelected>>")


def trainer_combobox_click_bind(e):
    trainer_name = trainer_combobox.get()
    joined_pokemon_list = " ".join(
        td.get_pokemon_from_trainer_name(trainer_name=trainer_name)
    )
    tab1_text.delete('1.0', END)
    print_string = f"The trainer {trainer_name} has the following pokemon:\n {joined_pokemon_list}\n"
    tab1_text.insert(INSERT, print_string)


def fight_flag_clicked():
    trainer_name = trainer_combobox.get()
    input_str = f"Printing out {trainer_name}'s flags:"
    input_str += td.check_flags_for_trainer(trainer_name)
    tab1_text.insert(END, input_str)

root = tb.Window(themename="solar")

root.title("Nuzli - Your Nuzlocke Companion")
root.geometry("1480x1000")

# Styles
my_tk_styles = tb.Style()
my_tk_styles.configure("primary.TButton", font=("Helvetica, 18"))

# Colors: Default, primary, secondary, success, info, warning, danger
# light dark
# Create Labels
# my_label = tb.Label(text="Papaya", font=("Helvetica", 28), bootstyle="primary")
# my_label.pack(pady=50)

# my_frame = tb.Frame(root, bootstyle="success")
# my_frame.pack(pady=40, font=("Helvetica", 14))

nuzi_book = tb.Notebook(root, bootstyle="dark")
# nuzi_book.place(relx=0.5, rely=0.5, anchor='center')
nuzi_book.place(relx=0.5, rely=0.5, anchor="center")

tab1 = tb.Frame(nuzi_book)  # , width=500, height=500)
# tab1.grid(row=4, column=4)
tab2 = tb.Frame(nuzi_book)  # , width=500, height=500)

tab1_tl = tb.Frame(tab1)  # , width=250, height=250)
tab1_tl.grid(column=0, row=0)
tab1_tr = tb.Frame(tab1)  # , width=250, height=250)
tab1_tr.grid(column=3, row=0)
tab1_bl = tb.Frame(tab1)  # , width=250, height=250)
tab1_bl.grid(column=0, row=3, columnspan=2, rowspan=2)
tab1_br = tb.Frame(tab1)  # , width=250, height=250)
tab1_br.grid(column=3, row=3)

# tab_label = Label(tab1, text="Ayaya", font=("Helvetica", 12))
# tab_label.pack(pady=20)

# my_text = Text(tab1, width=50, height=10)
# my_text.pack(pady=10, padx=10)


# my_label2 = Label(tab2, text="Tab 2!",font=("Helvetica",12))
# my_label2.pack(pady=20)

# Add our frames to the notebook
nuzi_book.add(tab1, text="Fights")
nuzi_book.add(tab2, text="Papaya")

# Create Buttons
# my_button = tb.Button(text="Clickie",
#                     bootstyle="success",
#                    style="primary.TButton",
#                   width=20)
# my_button.pack(pady=20)

tab1_title = Label(tab1_tl, text="Widget Placeholder", font=("Helvetica",18))
tab1_title.pack(pady=10)

# tab1_title = Label(tab1_tr, text = "Widget Placeholder)")
# tab1_title.pack(pady=5)
# tab1_text = Label(tab1_tr, text="papaya")
# tab1_text.pack(pady=5)

# Set Combo Default

# Create Split Combobox
split_combobox = tb.Combobox(tab1_br, bootstyle="success", values=td.get_splits())
split_combobox.pack(pady=5, padx=20)
# split_combobox.place(relx=0.8, rely=0.7, anchor='center')
split_combobox.set("Select a Split")


# binding the split combobox
split_combobox.bind("<<ComboboxSelected>>", split_combobox_click_bind)

# Create the trainer Combobox that displays the trainers from split
zone_combobox = tb.Combobox(tab1_br, bootstyle="success")
zone_combobox.pack(pady=5)
# zone_combobox.place(relx=0.8, rely=0.75, anchor='center')

# bind the trainer combobox
zone_combobox.bind("<<ComboboxSelected>>", zone_combobox_click_bind)

# Trainer combobox after seeing what zone we are

trainer_combobox = tb.Combobox(tab1_br, bootstyle="success")
trainer_combobox.pack(pady=5)
# trainer_combobox.place(relx=0.8, rely = 0.8, anchor='center')
# Bind the trainer combobox
trainer_combobox.bind("<<ComboboxSelected>>", trainer_combobox_click_bind)


fight_check_button = tb.Button(tab1_br, text="Check for flags", bootstyle="alert", 
                               command=fight_flag_clicked)
# fight_check_button.place(relx=0.5, rely=0.95, anchor='center')
fight_check_button.pack(pady=10)

tab1_text = ScrolledText(tab1_bl, height=25, width=50, wrap=WORD, autohide=True)
tab1_text.pack(pady=20, padx=20)
# tab1_text.grid(row=4, column=5)
# tab1_text.place(relx=0.5, rely=0.5, anchor='center')
root.mainloop()
