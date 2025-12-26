import ttkbootstrap as tb
import tkinter




level_list = ['Lu83', 'Lu30', 'Lu35', 'Lu33', 'Lu93', 'Luz']
hp_value_list = ['209/205', '130/266', '156/293', '178/255', '199/302', '76/76']
name_list = ['Shorki', 'Prina', 'Cerbi', 'feta', 'Heary', 'Celi', 'Cancel']


#Drop down options
game_splits = ['Grunt', 'Museum']
grunt_split = ['metronome', 'karpies']
museum_split = ['papaya', 'fishies', 'krab']

trainer_data = {
    'split_names': ['Choose current split','Grunt', 'Museum'],
    'Grunt': ['metronome', 'karpies'],
    'Museum': ['Jell', 'Fish']
}



#4 and 0 looks similar

if (name_list[-1] == 'Cancel'):
    name_list.pop()


root = tb.Window(themename="solar")

root.title("Title!")
root.geometry('500x350')

#Create Functions for button
def split_selected():
    my_label.config(text=f"You clicked on {split_combobox.get()}!")

#Create Binding Functions
def split_combobox_click_bind(e):
    trainer_combobox.config(values=trainer_data[split_combobox.get()])
    trainer_combobox.current(0)
    trainer_combobox.event_generate("<<ComboboxSelected>>")

def trainer_combobox_click_bind(e):
    my_label.config(text=f"{trainer_combobox.get()}")


#Styles
my_tk_styles = tb.Style()
my_tk_styles.configure('primary.TButton', 
                       font =("Helvetica, 18"))

#Colors: Default, primary, secondary, success, info, warning, danger
#light dark
#Create Labels
my_label = tb.Label(text = "Papaya", font=("Helvetica", 28), bootstyle="primary")
my_label.pack(pady=50)


#Create Buttons
#my_button = tb.Button(text="Clickie", 
 #                     bootstyle="success",
  #                    style="primary.TButton",
   #                   width=20)
#my_button.pack(pady=20)

#Set Combo Default

#Create Split Combobox
split_combobox = tb.Combobox(root, bootstyle="success",
                             values = trainer_data['split_names'])
split_combobox.pack(pady=20)
split_combobox.current(0)

#binding the split combobox
split_combobox.bind("<<ComboboxSelected>>",split_combobox_click_bind)

#Create the trainer Combobox that displays the trainers from split
trainer_combobox = tb.Combobox(root, bootstyle="success")
trainer_combobox.pack(pady=40)
#bind the trainer combobox
trainer_combobox.bind("<<ComboboxSelected>>", trainer_combobox_click_bind)

root.mainloop()


