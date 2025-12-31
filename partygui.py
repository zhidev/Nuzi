from tkinter import *  # noqa: F403

import ttkbootstrap as tb
from ttkbootstrap.scrolled import ScrolledText

import data_logic as td
# import image_process as nuzi_ocr
from snip_test import SnippingTool
from PIL import Image, ImageTk, ImageEnhance
import pygetwindow as gw
import pyautogui
import numpy as np


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


def adjust_photoimage_brightness(img, factor):
    # Convert ImageTk back to PIL Image
    pil_img = ImageTk.getimage(img)
    
    #  Adjust brightness
    # factor > 1.0 brightens; factor < 1.0 dims
    enhancer = ImageEnhance.Brightness(pil_img)
    brightened_pil = enhancer.enhance(factor)
    
    # Convert back to ImageTk.PhotoImage
    return ImageTk.PhotoImage(brightened_pil)

# def changing_zone_combobox(split_name):
#     zone_values=td.get_zones_in_split(split_name))

#     zone_combobox.config(values=zone_values)
#     zone_combobox.set(f"Select a Zone from the {split_name} split")
def disable_buttons_when_no_trainer():
    fight_check_button.config(state="disabled")
    tab3_fight_button.config(state="disabled")

def enable_buttons_when_trainer():
    fight_check_button.config(state="normal")
    tab3_fight_button.config(state="normal")

# Create Binding Functions
def split_combobox_click_bind(e):
    split_name = split_combobox.get()

    zone_combobox.config(values=td.get_zones_in_split(split_name))
    zone_combobox.set(f"Select a Zone from the {split_name} split")

    #update fight data tab text
    tab3_entry_split_text.set(split_name)
    tab3_entry_zone_text.set("Waiting Zone Data")
    tab3_entry_trainer_text.set("Waiting Trainer Data")
    disable_buttons_when_no_trainer()

def zone_combobox_click_bind(e):
    zone_name = zone_combobox.get()
    trainer_combobox.config(
        values=td.get_trainers_in_split_in_zone(
            split_name=split_combobox.get(), zone_name=zone_name
        )
    )
    tab3_entry_zone_text.set(zone_name)
    tab3_entry_trainer_text.set("Waiting Trainer Data")

    trainer_combobox.set(f"Select the trainer from the Zone: {zone_name}")
    # trainer_combobox.event_generate("<<ComboboxSelected>>")
    disable_buttons_when_no_trainer()


def trainer_combobox_click_bind(e):
    trainer_name = trainer_combobox.get()
    joined_pokemon_list = " ".join(
        td.get_pokemon_from_trainer_name(trainer_name=trainer_name)
    )
    tab1_text.delete("1.0", END)
    print_string = f"The trainer {trainer_name} has the following pokemon:\n {joined_pokemon_list}\n"
    tab1_text.insert(INSERT, print_string)
    tab3_entry_trainer_text.set(trainer_name)
    enable_buttons_when_trainer()    


def fight_flag_clicked():
    trainer_name = trainer_combobox.get()
    input_str = f"Printing out {trainer_name}'s flags:"
    input_str += td.check_flags_for_trainer(trainer_name)
    tab1_text.insert(END, input_str)


def screenshot_button_clicked():
    print("OCR Button Pressed")
    SnippingTool(root, callback=handle_image).start()
    pass


def handle_image(img):
    # We get here from ocr_button_clicked, and we recieves a PIL image
    print(f"Received image: {img.size}")
    #cv_img = nuzi_ocr.convert_PIL_to_cv_img(img)
    # print("Image converted")
    # img =Image.open("rnbimage.png")
    img = ImageTk.PhotoImage(img)
    print(f"Test image: {img}")

    ocr_img_label.config(image=img)
    ocr_img_label.image = img
    #print(nuzi_ocr.display(cv_img))
    #nuzi_ocr.ocr_image(pil_img)
    pass

def ocr_button_clicked():
    if hasattr(ocr_img_label, 'image') and ocr_img_label.image is not None:
        print("The label has an image reference.")
        input_image = ImageTk.getimage(ocr_img_label.image)
        # cv_img = td.convert_PIL_to_cv_img(input_image)
        print("Inside ocr button clicked")
        # td.ocr_image
        # nuzi_ocr.check_image_type(cv_img)
        # nuzi_ocr.ocr_image(cv_img)
        
        cv_img = td.convert_PIL_to_cv_img(input_image)
        
        if cv_img is None or cv_img.size == 0:
            print("Error: Converted image is empty!")
            return
        name_list = td.ocr_image_for_names(cv_img)
        tab3_party_text.delete("1.0", END)
        tab3_party_text.insert(INSERT, name_list)        
    else:
        print("The label does not have an image reference.")

def window_capture_and_display():
    # Find window
    windows = gw.getWindowsWithTitle("mGBA") # Grab windows with input text
    if not windows:
        print("Window not found")
        return
    
    win = windows[0] #grab first instance of the window name
    print(f"All the windows are: \n{windows}")
    region = (win.left, win.top, win.width, win.height)

    # Capture screenshot directly into a PIL object
    screenshot = pyautogui.screenshot(region=region)

    # Convert PIL image to Tkinter-compatible PhotoImage
    tk_image = ImageTk.PhotoImage(screenshot)

    # Update the label in your different function
    ocr_img_label.config(image=tk_image)
    ocr_img_label.image = tk_image  



def brighten_image():
    print("Brighten image")
    if hasattr(ocr_img_label, 'image') and ocr_img_label.image is not None:
        print("The label has an image reference.")
        input_image = adjust_photoimage_brightness(ocr_img_label.image, 1.15)
        ocr_img_label.config(image=input_image)
        ocr_img_label.image = input_image
    else:
        print("The label does not have an image reference.")

def dim_image():
    if hasattr(ocr_img_label, 'image') and ocr_img_label.image is not None:
        print("The label has an image reference.")
        input_image = adjust_photoimage_brightness(ocr_img_label.image, 0.85)
        ocr_img_label.config(image=input_image)
        ocr_img_label.image = input_image
    else:
        print("The label does not have an image reference.")

def submit_fight_data():
    pass


root = tb.Window(themename="darkly")

root.title("Nuzli - Your Nuzlocke Companion")
root.geometry("1400x800")

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
tab3 = tb.Frame(nuzi_book)

tab1_tl = tb.Frame(tab1)
tab1_tl.grid(column=0, row=0)
tab1_tr = tb.Frame(tab1)
tab1_tr.grid(column=3, row=0)
tab1_bl = tb.Frame(tab1)
tab1_bl.grid(column=0, row=3, columnspan=2, rowspan=2)
tab1_br = tb.Frame(tab1)
tab1_br.grid(column=3, row=3)

# Add our frames to the notebook
nuzi_book.add(tab1, text="Fights")
nuzi_book.add(tab2, text="Papaya")
nuzi_book.add(tab3, text="Ocr")

# Create Buttons
# my_button = tb.Button(text="Clickie",
#                     bootstyle="success",
#                    style="primary.TButton",
#                   width=20)
# my_button.pack(pady=20)

tab1_title = Label(tab1_tl, text="Widget Placeholder", font=("Helvetica", 18))
tab1_title.pack(pady=10)


# Set Combo Default

# Create Split Combobox
split_combobox = tb.Combobox(tab1_br, bootstyle="success", values=td.get_splits())
split_combobox.pack(pady=5, padx=20)
split_combobox.set("Select a Split")


# binding the split combobox
split_combobox.bind("<<ComboboxSelected>>", split_combobox_click_bind)

# Create the trainer Combobox that displays the trainers from split
zone_combobox = tb.Combobox(tab1_br, bootstyle="success")
zone_combobox.pack(pady=5)

# bind the trainer combobox
zone_combobox.bind("<<ComboboxSelected>>", zone_combobox_click_bind)

# Trainer combobox after seeing what zone we are

trainer_combobox = tb.Combobox(tab1_br, bootstyle="success")
trainer_combobox.pack(pady=5)

# Bind the trainer combobox
trainer_combobox.bind("<<ComboboxSelected>>", trainer_combobox_click_bind)


fight_check_button = tb.Button(
    tab1_br, text="Check for flags", bootstyle="alert", command=fight_flag_clicked,
    state="disabled"
)
fight_check_button.pack(pady=10)

tab1_text = ScrolledText(tab1_bl, height=25, width=50, wrap=WORD, autohide=True)
tab1_text.pack(pady=20, padx=20)


#
# Tab 2 Stuff for screenshot processing
#

tab2_tl = tb.Frame(tab2)
tab2_tl.grid(column=0, row=0)
tab2_tr = tb.Frame(tab2)
tab2_tr.grid(column=3, row=0)
tab2_bl = tb.Frame(tab2)
tab2_bl.grid(column=0, row=3, columnspan=2, rowspan=2)
tab2_br = tb.Frame(tab2)
tab2_br.grid(column=3, row=3)


ocr_instruction_text = "Please only Capture the 6 Party members\n" \
"Then proceed to the next tab when satisfactory\nBrighten the image if neccesary"
ocr_instruction_label = Label(tab2_tl, text=ocr_instruction_text, font=("Helvetica", 12))
ocr_instruction_label.pack(pady=10)

screenshot_button = tb.Button(
    tab2_br, text="Get Picutre of Team", bootstyle="alert", command=screenshot_button_clicked
)
screenshot_button.pack(pady=10)

window_capture_button = tb.Button(
    tab2_br, text="Window Capture mGBA", bootstyle="alert", 
      command=window_capture_and_display)
window_capture_button.pack(pady=5)

# test_img =Image.open("rnbimage.png")
# test_img = ImageTk.PhotoImage(test_img)
ocr_img_label = Label(tab2_bl, bd=2, relief="raised")
ocr_img_label.pack(pady=5)


ocr_button = tb.Button(
    tab2_br, text="Ocr the picture", bootstyle="alert", command=ocr_button_clicked
)
ocr_button.pack(pady=10)

brighten_button = tb.Button(
    tab2_bl, text="Brighten Image", bootstyle="Success", command=brighten_image
)
brighten_button.pack(side=LEFT,pady=5)

dim_button = tb.Button(
    tab2_bl, text="Dim Image", bootstyle="Failure", command=dim_image
)
dim_button.pack(side=LEFT,pady=5)

#
# Tab 3 stuff for ocr and team entry
#

tab3_tl = tb.Frame(tab3)
tab3_tl.grid(column=0, row=0)
tab3_tr = tb.Frame(tab3)
tab3_tr.grid(column=2, row=0)
tab3_bl = tb.Frame(tab3)
tab3_bl.grid(column=0, row=2, columnspan=2, rowspan=2)
tab3_br = tb.Frame(tab3)
tab3_br.grid(column=2, row=2)


tab3_label = Label(tab3_tl, text="Logging Fight Data", font=("Helvetica", 12))
tab3_label.pack(pady=5)
tab3_instruction_text = "Please Fix any discrepency from the ocr and follow the format\n" \
"[Pokemon Name] [Pokemon2 Name] [Pokemon3 Name]\n[Pokemon4 Name] [Pokemon5 Name] [Pokemon6 Name]\n" \
"Comments:\n Comments in the bottom box, seperate Pokemons in brackets and space"


tab3_instructions = Label(tab3_tl, text=tab3_instruction_text, font=("Helvetica", 10))
tab3_instructions.pack(pady=5)

tab3_party_text = ScrolledText(tab3_bl, height=5, width=30, wrap=WORD, autohide=True)
tab3_party_text.pack(pady=20, padx=20)

tab3_comment_text = ScrolledText(tab3_bl, height=10, width=30, wrap=WORD, autohide=True)
tab3_comment_text.pack(pady=20, padx=20)


#Entry Display Information
tab3_entry_split_text = tb.StringVar(value="Waiting Split Data")
tab3_entry_split = tb.Entry(tab3_br,textvariable= tab3_entry_split_text,
                            state="readonly")
tab3_entry_split.pack(pady=5)

tab3_entry_zone_text = tb.StringVar(value="Waiting Zone Data")
tab3_entry_zone = tb.Entry(tab3_br,textvariable= tab3_entry_zone_text,
                            state="readonly")
tab3_entry_zone.pack(pady=5)

tab3_entry_trainer_text = tb.StringVar(value="Waiting Trainer Data")
tab3_entry_trainer = tb.Entry(tab3_br,textvariable= tab3_entry_trainer_text,
                            state="readonly")
tab3_entry_trainer.pack(pady=5)




tab3_fight_button = tb.Button(
    tab3_br, text="Save Fight ", 
    bootstyle="primary", command=submit_fight_data, state="disabled"
)
tab3_fight_button.pack(padx= 35,pady=5)





root.mainloop()
