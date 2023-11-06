import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# region global var

selected_class = []
selected_feature = []
bias = 0
label_encoder = LabelEncoder()
scaler = StandardScaler()

# endregion


def windwos():

    # region Forms

    def form1():

        # check the value and run main
        def launch():
            try:
                if (len(selected_class[-1]) == 2 & len(selected_feature[-1]) == 2):

                    val = text_box.get("1.0", "end-1c")

                    if (radio_var.get() == "yes"):
                        try:
                            val = int(val)
                        except:
                            messagebox.showerror("Error", "Bias should be a Number")
                            text_box.delete("1.0", "end")
                            restart()
                    else:
                        val = 0

                    root.destroy()
                else:
                    messagebox.showerror("Error", "Requirements not met")


            except:
                messagebox.showerror("Error", "!!احا")

            main(selected_class[-1], selected_feature[-1], val)

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
                checkbox1 = tk.Checkbutton(root, text=choice, variable=var1, font=("Helvetica", 12), padx=5, pady=3,
                                           command=show_selection_class)
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
                checkbox = tk.Checkbutton(root, text=choice, variable=var, font=("Helvetica", 12), padx=5, pady=3,
                                          command=show_selectioncombo)
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
                    text_box.place(x="110", y="452")
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

        infolabel1 = tk.Label(text="Choose Two Classes", font=("Times New Roman", 18), fg="blue", pady=8)
        infolabel1.pack()

        avalible_class = ["BOMBAY", "CALI", "SIRA"]

        checkboxes1 = create_class_checkboxes(avalible_class)

        result_label1 = tk.Label(root, text="", font=("Helvetica", 12))
        result_label1.pack()

        # endregion

        # region seperated line
        seperated_line1 = tk.Label(root, text="----------------------------------------------------")
        seperated_line1.pack()
        # endregion

        # region choose 2 features

        infolabel2 = tk.Label(text="Choose Two Features", font=("Times New Roman", 18), fg="blue", pady=8)
        infolabel2.pack()

        avalible_choices = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]

        checkboxes = create_checkboxes(avalible_choices)

        result_label = tk.Label(root, text="", font=("Helvetica", 12))
        result_label.pack()

        # endregion

        # region seperated line
        seperated_line2 = tk.Label(root, text="----------------------------------------------------")
        seperated_line2.pack()
        # endregion

        # region choose bias or not

        radio_var = tk.StringVar()

        radio_button1 = tk.Radiobutton(root, text="Bias", variable=radio_var, value="yes", font=("Helvetica", 12),
                                       command=show_textbox)
        radio_button1.pack(anchor="w")

        radio_button2 = tk.Radiobutton(root, text="No Bias", variable=radio_var, value="no", font=("Helvetica", 12),
                                       command=show_textbox)
        radio_button2.pack(anchor="w")

        text_box = tk.Text(root, height=1, width=18, font=("Helvetica", 12))

        show_text = tk.IntVar()
        show_text.set(0)

        # endregion

        # region seperated line
        seperated_line3 = tk.Label(root, text="----------------------------------------------------")
        seperated_line3.pack()
        # endregion

        # region Program button

        prog_button = tk.Button(root, text="Next", font=("Times New Roman", 15, "bold"), command=launch)
        prog_button.place(x="99", y="540")

        # endregion

        root.mainloop()

    def form2():

        def launch():

            Learning = Learning_Text.get("1.0", "end-1c")
            epochs = epochs_Text.get("1.0", "end-1c")
            MSE = MSE_Text.get("1.0", "end-1c")

            try:

                Learning = float(Learning)
                epochs = int(epochs)
                MSE = float(MSE)

                global return_values
                return_values = (
                    Learning,
                    epochs,
                    MSE,
                    radio_var.get()
                )

                root.destroy()
            except:

                messagebox.showerror("Error", "Enter Numbers")

        # region Base of Tkinter Window

        root = tk.Tk()
        root.title("Task1")
        root.iconbitmap("images\FCIS.ico")
        root.geometry("300x600")
        root.resizable(width=False, height=False)

        # endregion

        # region Learning Rate
        Learning_Label = tk.Label(text="Learning Rate", font=("Times New Roman", 12), fg="blue", pady=8)
        Learning_Label.pack(anchor='w')

        Learning_Text = tk.Text(root, height=1, width=18, font=("Helvetica", 12))
        Learning_Text.pack()
        # endregion

        # region seperated line
        seperated_line1 = tk.Label(root, text="----------------------------------------------------")
        seperated_line1.pack()
        # endregion

        # region epochs
        epochs_Label = tk.Label(text="Number of Epochs", font=("Times New Roman", 12), fg="blue", pady=8)
        epochs_Label.pack(anchor='w')

        epochs_Text = tk.Text(root, height=1, width=18, font=("Helvetica", 12))
        epochs_Text.pack()
        # endregion

        # region seperated line
        seperated_line1 = tk.Label(root, text="----------------------------------------------------")
        seperated_line1.pack()
        # endregion

        # region MSE
        MSE_Label = tk.Label(text="MSE threshold ", font=("Times New Roman", 12), fg="blue", pady=8)
        MSE_Label.pack(anchor='w')

        MSE_Text = tk.Text(root, height=1, width=18, font=("Helvetica", 12))
        MSE_Text.pack()
        # endregion

        # region seperated line
        seperated_line1 = tk.Label(root, text="----------------------------------------------------")
        seperated_line1.pack()
        # endregion

        # region Choose Algorithm

        radio_var = tk.StringVar()

        radio_button1 = tk.Radiobutton(root, text="Perceptron Algo", variable=radio_var, value="Perceptron", font=("Helvetica", 12))
        radio_button1.pack(anchor="w")

        radio_button2 = tk.Radiobutton(root, text="AdaLine Algo", variable=radio_var, value="AdaLine", font=("Helvetica", 12))
        radio_button2.pack(anchor="w")

        # endregion

        # region start_Program button

        prog_button = tk.Button(root, text="Launch", font=("Times New Roman", 15, "bold"), command=launch)
        prog_button.place(x="99", y="540")

        # endregion

        root.mainloop()

    # endregion

    def PreProcessing(SelectedClass, SelectedFeature):
        data = pd.read_csv("Dry_Beans_Dataset.csv")  # read file
        class_data = data[data['Class'].isin(SelectedClass)]  # take two classes
        data = class_data[SelectedFeature]  # take two features only
        data['Class'] = class_data.iloc[:,-1] # add y column

        if(SelectedFeature[0] != "Area"):  # convert from string to float
            data[SelectedFeature[0]] = data[SelectedFeature[0]].str.replace("٫", ".").astype(float)
            data[SelectedFeature[1]] = data[SelectedFeature[1]].str.replace("٫", ".").astype(float)
        else:
            data[SelectedFeature[1]] = data[SelectedFeature[1]].str.replace("٫", ".").astype(float)

        data = data.fillna(data.mean())  # fillna

        Class1 = data.iloc[:50, :] # seperate data and take first class
        Class2 = data.iloc[50:, :] # seperate data and take second class

        x_train1, x_test1, y_train1, y_test1 = train_test_split(Class1.iloc[:, :-1].values, Class1.iloc[:, -1].values, test_size = 20, random_state = 200) # 30/20
        x_train2, x_test2, y_train2, y_test2 = train_test_split(Class2.iloc[:, :-1].values, Class2.iloc[:, -1].values, test_size = 20, random_state = 200) # 30/20

        x_train1 = pd.DataFrame(x_train1, columns=Class1.iloc[:, :-1].columns)
        x_test1 = pd.DataFrame(x_test1, columns=Class1.iloc[:, :-1].columns)
        y_train1 = pd.DataFrame(y_train1, columns=[Class1.columns[-1]])
        y_test1 = pd.DataFrame(y_test1, columns=[Class1.columns[-1]])

        x_train2 = pd.DataFrame(x_train2, columns=Class2.iloc[:, :-1].columns)
        x_test2 = pd.DataFrame(x_test2, columns=Class2.iloc[:, :-1].columns)
        y_train2 = pd.DataFrame(y_train2, columns=[Class1.columns[-1]])
        y_test2 = pd.DataFrame(y_test2, columns=[Class1.columns[-1]])

        X_TRAIN1 = pd.concat([x_train1, x_train2], axis=0)
        X_TRAIN = scaler.fit_transform(X_TRAIN1)
        X_TRAIN = pd.DataFrame(X_TRAIN, columns=X_TRAIN1.columns)

        X_TEST1 = pd.concat([x_test1, x_test2], axis=0)
        X_TEST = scaler.fit_transform(X_TEST1)
        X_TEST = pd.DataFrame(X_TEST, columns=X_TEST1.columns)

        Y_TRAIN = pd.concat([y_train1, y_train2], axis=0)
        Y_TRAIN['Class'] = label_encoder.fit_transform(Y_TRAIN)

        Y_TEST = pd.concat([y_test1, y_test2], axis=0)
        Y_TEST['Class'] = label_encoder.fit_transform(Y_TEST)

        return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

    # region Graph
    def visualize_decision_boundary(Weight, X_TEST, Y_TEST):
        fig, ax = plt.subplots(figsize=(8, 6))

        X_TEST_reset = X_TEST.reset_index(drop=True)
        Y_TEST_reset = Y_TEST.reset_index(drop=True)

        ax.scatter(X_TEST_reset.iloc[:, 0], X_TEST_reset.iloc[:, 1], c=Y_TEST_reset['Class'], cmap='coolwarm',
                   label='Data Points')

        if len(Weight) == 2:  # For 2D classification with two weights (1 bias + 1 weight for 2D)
            bias, weight1 = Weight[0], Weight[1]
            x_values = np.linspace(X_TEST_reset.iloc[:, 0].min(), X_TEST_reset.iloc[:, 0].max(), 100)
            y_values = (-bias - weight1 * x_values) / weight1
        elif len(Weight) == 3:  # For 2D classification with three weights (1 bias + 2 weights for 2D)
            bias, weight1, weight2 = Weight[0], Weight[1], Weight[2]
            x_values = np.linspace(X_TEST_reset.iloc[:, 0].min(), X_TEST_reset.iloc[:, 0].max(), 100)
            y_values = (-bias - weight1 * x_values) / weight2

        # Plot the decision boundary line
        ax.plot(x_values, y_values, label='Decision Boundary', color='green')

        ax.set_xlabel(X_TEST_reset.columns[0])
        ax.set_ylabel(X_TEST_reset.columns[1])
        ax.legend()
        ax.set_title('Decision Boundary and Test Data Distribution')

        plt.show()

    # endregion

    # region Perceptron
    def Perceptron(Weight, X_TRAIN, Y_TRAIN, learning_rate, num_epochs):

        def signum(x):
            if(x > 0):
                return 1
            elif(x == 0):
                return 0
            elif(x < 0):
                return -1

        X_TRAIN = X_TRAIN.to_numpy()
        for epoch in range(num_epochs):

            for i in range(len(X_TRAIN)):
                predict = signum(np.dot(Weight.T, X_TRAIN[i]))
                if (predict != Y_TRAIN.iloc[i,0]):
                    loss = Y_TRAIN.iloc[i,0] - predict
                    Weight += learning_rate * loss * X_TRAIN[i]

        return Weight

    def Perceptron_Test(Weight, X_TEST, Y_TEST):

        def signum(x):
            if x > 0:
                return 1
            elif x == 0:
                return 0
            elif x < 0:
                return -1

        m = len(Y_TEST)
        wrong = 0
        y_pred = X_TEST.dot(Weight.transpose())

        # Convert Pandas Series to NumPy arrays for element-wise comparison
        yPredTest = y_pred.apply(signum).to_numpy()

        Y_TEST = Y_TEST.to_numpy()

        for i in range(m):
            if yPredTest[i] != Y_TEST[i]:
                wrong += 1

        Accuracy = ( ( (m-wrong) / m ) * 100)

        return Accuracy,yPredTest

    # endregion

    # region AdaLine
    def AdaLine(Weight, X_TRAIN, Y_TRAIN, learning_rate, num_epochs, MSE):

        m = len(X_TRAIN)
        for epoch in range(num_epochs):
            for i in range(len(X_TRAIN)):
                y_pred = np.dot(Weight.T, X_TRAIN.iloc[i])
                loss = Y_TRAIN.iloc[i,0] - y_pred
                Weight += learning_rate * loss * X_TRAIN.to_numpy()[i]

            predictions = np.dot(X_TRAIN, Weight.transpose())
            mse = 1 / (2 * m) * np.sum((Y_TRAIN.to_numpy() - predictions) ** 2)

            if mse <= MSE:
                print(f"Training stopped at epoch {epoch} because MSE reached the threshold.")
                break
        return Weight

    def AdaLine_Test(Weight, X_TEST, Y_TEST):

        prediction = X_TEST.dot(Weight.transpose())
        mse = 1/(2*len(X_TEST)) * np.sum((Y_TEST.to_numpy() - prediction.to_numpy())**2)
        return mse,prediction

    # endregion

    def main(SelectedClass, SelectedFeature, BiasVal):

        def confusion_matrix(true_labels, predicted_labels):
            true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
            for i in range(len(predicted_labels)):
                true_label = true_labels.iloc[i, 0]
                predicted_label = predicted_labels[i]
                if true_label == 1 and predicted_label == 1:
                    true_positive += 1
                elif true_label == 0 and predicted_label == 0:
                    true_negative += 1
                elif true_label == 0 and predicted_label == 1:
                    false_positive += 1
                elif true_label == 1 and predicted_label == 0:
                    false_negative += 1

            confusion_matrix = [[true_negative, false_positive], [false_negative, true_positive]]

            return confusion_matrix

        # region get the Data

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = PreProcessing(SelectedClass, SelectedFeature)

        # endregion

        # region (W)RandomNumber

        if(BiasVal != 0):
            W0 = BiasVal
            W1 = random.random()
        else:
            W0 = random.random()
            W1 = random.random()

        Weight = np.array([W0, W1])



        #endregion

        # region get values for algo

        form2()
        if return_values is not None:
            learning_rate, num_epochs, MSE, Algo = return_values
        # endregion

        # region Algo
        prediction = 0
        if(Algo == "Perceptron"):
            new_Weight = Perceptron(Weight, X_TRAIN, Y_TRAIN, learning_rate, num_epochs)
            Accuracy,prediction = Perceptron_Test(new_Weight, X_TEST, Y_TEST)
            print(f"Perceptron Accuracy =  {Accuracy} % ")
        else:
            new_Weight = AdaLine(Weight, X_TRAIN, Y_TRAIN, learning_rate, num_epochs, MSE)
            Accuracy,prediction = AdaLine_Test(new_Weight, X_TEST, Y_TEST)
            print(f"AdaLine Accuracy =  {Accuracy} % ")

        # endregion

        # region Confusion Matrix
        matrix = confusion_matrix(Y_TEST, prediction)
        print("confusion_matrix", matrix)
        # endregion

        # region Graph

        visualize_decision_boundary(new_Weight, X_TEST, Y_TEST)

        # endregion

    form1()

warnings.filterwarnings("ignore")
windwos()
print("End Program")
