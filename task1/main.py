import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd

# region global var

selected_class = []
selected_feature = []
bias = 0

# endregion


def windwos():

    def PreProcessing(SelectedClass, SelectedFeature, BiasVal):
        data = pd.read_csv("Dry_Beans_Dataset.csv")  # read file
        x = data[SelectedFeature]  # take two features only
        print(x.head())

    # Pass the result and open new window
    def bridge():
        try:

            if( len(selected_class[-1]),len(selected_feature[-1]) == 2):

                # # you can get the selected feature by the variable selected_feature
                # print("selected class: ", selected_class[-1], "\n")
                # # you can get the selected feature by the variable selected_feature
                # print("selected feature: ", selected_feature[-1],"\n")
                # # you can know bias or not by variable radio_var.get()
                # print("bias1 or not0: ", radio_var.get(),"\n")
                # you can know the number of bias using text_box.get("1.0", "end-1c")
                val = text_box.get("1.0", "end-1c")

                if(radio_var.get() == "yes"):
                    try:
                        val = int(val)
                    except:
                        messagebox.showerror("Error", "Bias should be a Number")
                        text_box.delete("1.0", "end")
                else:
                    val = 0
                # print("textbox val: ", val,"\n")

                root.destroy()
                # call another window
            else:
                messagebox.showerror("Error", "Requirements not met")


        except:
            messagebox.showerror("Error", "!!احا")

        PreProcessing(selected_class[-1], selected_feature[-1], val)

    # using to create a new window
    def restart():
        root.destroy()  # Close the current main window
        windwos()

    # region Class Functions

    # appear label or not
    def show_selection_class():
        selected_choices1 = [choice for choice, var1 in checkboxes1.items() if var1.get()]

        if len(selected_choices1) > 2:
            result_label1.config(text="Please select exactly two Classes.", fg="red")
            selected_class.append(selected_choices1)
        else:
            result_label1.config(text="")
            selected_class.append(selected_choices1)

    # get the selected one
    def create_class_checkboxes(avalible_class):
        checkboxes_list1 = {}
        for choice in avalible_class:
            var1 = tk.BooleanVar()
            checkbox1 = tk.Checkbutton(root, text=choice, variable=var1, font=("Helvetica", 12) , padx=5, pady=3, command=show_selection_class)
            checkbox1.pack(anchor="w")
            checkboxes_list1[choice] = var1
        return checkboxes_list1

    # endregion

    # region Features Functions

    # using with feature to appear label or not
    def show_selectioncombo():
        selected_choices = [choice for choice, var in checkboxes.items() if var.get()]

        if len(selected_choices) > 2:
            result_label.config(text="Please select exactly two Features.", fg="red")
            selected_feature.append(selected_choices)
        else:
            result_label.config(text="")
            selected_feature.append(selected_choices)

    # using with feature to get the selected one
    def create_checkboxes(avalible_choices):
        checkboxes_list = {}
        for choice in avalible_choices:
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(root, text=choice, variable=var, font=("Helvetica", 12) , padx=5, pady=3,command=show_selectioncombo)
            checkbox.pack(anchor="w")
            checkboxes_list[choice] = var
        return checkboxes_list

    # endregion

    # using with bias
    def show_textbox():

        def clear_text(event):
            text_box.delete("1.0", "end")

        if (radio_var.get() == "yes"):
            if (show_text.get() == 0):
                text_box.place(x="110",y="452")
                text_box.bind("<Button-1>", clear_text)
                text_box.insert("2.0", "Enter the value of Bias")
                show_text.set(1)
                radio_button1.config(state="disabled")
                radio_button2.config(state="disabled")

        else:
            radio_button1.config(state="disabled")
            radio_button2.config(state="disabled")

    # region Base of Tkinter Window
    root = tk.Tk()
    root.title("Task1")
    root.iconbitmap("images\FCIS.ico")
    root.geometry("300x600")
    root.resizable(width=False, height=False)
    # endregion

    # region restart button
    original_image = Image.open("images\Arestart.png")
    resized_image = original_image.resize((25, 25), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(resized_image)
    button = tk.Button(root, image=image, command=restart)
    button.place(x="267", y="3")

    # endregion

    # region choose 2 class

    infolabel1 = tk.Label(text="Choose Two Classes",font=("Times New Roman", 18),fg="blue",pady=8)
    infolabel1.pack()

    avalible_class = ["BOMBAY", "CALI", "SIRA"]

    checkboxes1 = create_class_checkboxes(avalible_class)

    result_label1 = tk.Label(root, text="",font=("Helvetica", 12))
    result_label1.pack()

    # endregion

    # region seperated line
    seperated_line1 = tk.Label(root, text="----------------------------------------------------")
    seperated_line1.pack()
    # endregion

    # region choose 2 features

    infolabel2 = tk.Label(text="Choose Two Features",font=("Times New Roman", 18),fg="blue",pady=8)
    infolabel2.pack()

    avalible_choices = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]

    checkboxes = create_checkboxes(avalible_choices)

    result_label = tk.Label(root, text="",font=("Helvetica", 12))
    result_label.pack()

    # endregion

    # region seperated line
    seperated_line2 = tk.Label(root, text="----------------------------------------------------")
    seperated_line2.pack()
    # endregion

    # region choose bias or not

    radio_var = tk.StringVar()

    radio_button1 = tk.Radiobutton(root, text="Bias", variable=radio_var, value="yes",font=("Helvetica", 12), command=show_textbox)
    radio_button1.pack(anchor="w")

    radio_button2 = tk.Radiobutton(root, text="No Bias", variable=radio_var, value="no",font=("Helvetica", 12), command=show_textbox)
    radio_button2.pack(anchor="w")

    text_box = tk.Text(root, height=1, width=18,font=("Helvetica", 12))

    show_text = tk.IntVar()
    show_text.set(0)

    # endregion

    # region seperated line
    seperated_line3 = tk.Label(root, text="----------------------------------------------------")
    seperated_line3.pack()
    # endregion

    # region start_Program button

    prog_button = tk.Button(root, text= "Launch",font=("Times New Roman", 15, "bold"), command=bridge)
    prog_button.place(x="99",y="540")

    # endregion

    root.mainloop()


windwos()
print("End Program")
