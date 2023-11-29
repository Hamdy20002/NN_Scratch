import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class NN:

    neo_num_layer = []
    neo_num_layer_values = [3,3]
    bias = 0
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    parameters = {}
    actication = "Sigmoid"
    Biasval = 0
    epochs = 100
    Learning = 0.1

    def run(self):

        def form():

            # check the value and run main
            def launch():
                try:
                    self.neo_num_layer_values.clear()
                    for i in self.neo_num_layer:
                        self.neo_num_layer_values.append(int(i.get()))

                    # activation value
                    self.actication = radio_var.get()

                    # LEARNING RATE
                    self.Learning = float(Learning_Text.get("1.0", "end-1c"))
                    # EPOCS
                    self.epochs = int(epochs_Text.get("1.0", "end-1c"))
                    # BIAS VAL
                    if (checkbox_var.get() == 1):
                        self.Biasval = int(bias_text_box.get("1.0", "end-1c"))
                    else:
                        self.Biasval = 0

                    root.destroy()

                except:
                    messagebox.showerror("Error", "Enter Numbers Only!!")

            # using to create a new window
            def restart():
                root.destroy()  # Close the current main window
                self.run()

            # using with bias
            def show_textbox():

                def clear_text(event):
                    bias_text_box.delete("1.0", "end")

                if (checkbox_var.get() == 1):
                    bias_text_box.pack(anchor="w")
                    bias_text_box.bind("<Button-1>", clear_text)
                    bias_text_box.insert("2.0", "Enter the value of Bias")
                    checkbox1.configure(state="disabled")

            # using with create_textbox
            def create_dynamic_textboxes(event):

                try:
                    yy = 0
                    num_layers = int(Hidden_Layers_num.get("1.0", "end-1c"))
                    Hidden_Layers_num.config(state='disabled')
                    root.geometry("600x600")
                    Choose_Label = tk.Label(text="Enter The Number of Neurons In Each Hidden_Layers",
                                            font=("Times New Roman", 10), fg="blue", pady=8)
                    Choose_Label.place(x="300", y=str(yy))
                    for i in range(num_layers):
                        yy += 60
                        tk.Label(root, text=f"Neo num in layer{i + 1}", font=("Helvetica", 12)).place(x="380",
                                                                                                      y=str(yy - 25))
                        neo_num_entry = tk.Entry(root)
                        neo_num_entry.place(x="380", y=str(yy))
                        self.neo_num_layer.append(neo_num_entry)

                    textboxes_created = True
                except:
                    messagebox.showerror("Error", "Please enter a valid number for Hidden Layers")

            # region Base of Tkinter Window
            root = tk.Tk()
            root.title("Task2")
            root.iconbitmap("images\FCIS.ico")
            root.geometry("280x600")
            root.resizable(width=False, height=False)
            # endregion

            # region restart button
            original_image = Image.open("images\Arestart.png")
            resized_image = original_image.resize((25, 25), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(resized_image)
            button = tk.Button(root, image=image, command=restart)
            button.place(x="235", y="5")

            # endregion

            # region Number of Hidden Layers

            Hidden_Layers_Label = tk.Label(text="Numbers Of Hidden Layers", font=("Times New Roman", 12), fg="blue",
                                           pady=8)
            Hidden_Layers_Label.pack(anchor='w')

            Hidden_Layers_num = tk.Text(root, height=1, width=18, font=("Helvetica", 12))
            Hidden_Layers_num.pack(anchor='w')

            # endregion

            # region seperated line
            seperated_line1 = tk.Label(root, text="----------------------------------------------------")
            seperated_line1.pack(anchor='w')
            # endregion

            # region Choose Activation Function

            Choose_Label = tk.Label(text="Choose The Activation Function", font=("Times New Roman", 15), fg="blue",
                                    pady=8)
            Choose_Label.pack(anchor='w')

            radio_var = tk.StringVar()

            radio_button1 = tk.Radiobutton(root, text="Sigmoid", variable=radio_var, value="Sigmoid",
                                           font=("Helvetica", 12))
            radio_button1.pack(anchor="w")

            radio_button2 = tk.Radiobutton(root, text="Tangent", variable=radio_var, value="Tangent",
                                           font=("Helvetica", 12))
            radio_button2.pack(anchor="w")

            # endregion

            # region seperated line
            seperated_line1 = tk.Label(root, text="----------------------------------------------------")
            seperated_line1.pack(anchor='w')
            # endregion

            # region Learning Rate
            Learning_Label = tk.Label(text="Learning Rate", font=("Times New Roman", 12), fg="blue", pady=8)
            Learning_Label.pack(anchor='w')

            Learning_Text = tk.Text(root, height=1, width=18, font=("Helvetica", 12))
            Learning_Text.pack(anchor='w')
            # endregion

            # region seperated line
            seperated_line1 = tk.Label(root, text="----------------------------------------------------")
            seperated_line1.pack(anchor='w')
            # endregion

            # region epochs
            epochs_Label = tk.Label(text="Number of Epochs", font=("Times New Roman", 12), fg="blue", pady=8)
            epochs_Label.pack(anchor='w')

            epochs_Text = tk.Text(root, height=1, width=18, font=("Helvetica", 12))
            epochs_Text.pack(anchor='w')
            # endregion

            # region seperated line
            seperated_line1 = tk.Label(root, text="----------------------------------------------------")
            seperated_line1.pack(anchor='w')
            # endregion

            # region choose bias or not

            checkbox_var = tk.IntVar()

            # Create checkboxes
            checkbox1 = tk.Checkbutton(root, text="Bias", variable=checkbox_var, font=("Helvetica", 12),
                                       command=show_textbox)
            checkbox1.pack(anchor="w")

            bias_text_box = tk.Text(root, height=1, width=18, font=("Helvetica", 12))

            # endregion

            # region neo number for each layer

            Hidden_Layers_num.bind("<FocusOut>", create_dynamic_textboxes)

            # endregion

            # region vert seperated line
            yy = -50
            for i in range(20):
                yy += 30
                tk.Label(root, text="|").place(x="275", y=str(yy))

            # endregion

            # region Program button

            prog_button = tk.Button(root, text="Launch", font=("Times New Roman", 15, "bold"), command=launch)
            prog_button.place(x="80", y="500")

            # endregion

            root.mainloop()

        form()

    def fit(self):

        def forward_prop(input, parameters, Activation):

            # region Activation fun

            def sigmoid(z):
                s = 1 / (1 + np.exp(-z))
                return s

            def tanh(z):
                t = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
                return t

            # endregion

            hidden_layers = len(self.neo_num_layer_values)

            layers_outputs = {'input_layer': input}


            for i in range(1, hidden_layers + 2):

                if(i == 1):
                    layer_netval = np.dot(layers_outputs['input_layer'], parameters['w'+str(i)]) + parameters['b'+str(i)].T
                else:
                    layer_netval = np.dot(layers_outputs['hidden_layer'+str(i-1)], parameters['w'+str(i)]) + parameters['b'+str(i)].T


                saver = 0

                if (Activation == "Sigmoid"):
                    saver = sigmoid(layer_netval)
                elif (Activation == "Tanh"):
                    saver = tanh(layer_netval)

                if (i != hidden_layers + 1):
                    layers_outputs['hidden_layer' + str(i)] = saver
                elif (i == hidden_layers + 1):
                    layers_outputs['output_layer'] = saver

            return layers_outputs

        def backward_prop(input, targets, layers_outputs,parameters, learning_rate, Activation):

            # region activation function

            def sigmoid_derivative(output_layer):
                sigmoid_x = 1 / (1 + np.exp(-output_layer))
                return sigmoid_x * (1 - sigmoid_x)
            def tanh_derivative(output_layer):
                return 1 - np.tanh(output_layer) ** 2

            # endregion

            encoder = OneHotEncoder(sparse=False)
            y_train_one_hot = encoder.fit_transform(targets)

            errors = y_train_one_hot - layers_outputs['output_layer']

            if (Activation == "Sigmoid"):
                output_error = errors * sigmoid_derivative(layers_outputs["output_layer"])
                parameters["w"+str(len(layers_outputs) - 1)] += learning_rate * np.dot(layers_outputs['hidden_layer'+str(len(layers_outputs) - 2)].T, output_error)
                sum = np.sum(output_error, axis=0)
                parameters["b" + str(len(layers_outputs) - 1)] += learning_rate * sum.reshape((3, 1))
                GGG = output_error
                for i in range(len(layers_outputs) - 2, 0, -1):
                    print("Start")
                    if(i == 1):
                        print("Start SMALL")
                        hidden_error = np.dot(GGG, parameters["w1"].T) * sigmoid_derivative(layers_outputs["hidden_layer1"])
                        parameters["w1"] += learning_rate * np.dot(layers_outputs["input_layer"].T, layers_outputs["hidden_layer1"])
                        sum = np.sum(hidden_error, axis=0)
                        parameters["b1"] += learning_rate * sum.reshape((3, 1))
                        print("Finish SMALL")
                        break

                    hidden_error = np.dot(GGG, parameters["w" + str(i)].T) * sigmoid_derivative(layers_outputs["hidden_layer" + str(i)])
                    GGG = hidden_error
                    parameters["w" + str(i)] += (learning_rate * np.dot(hidden_error.T , layers_outputs['hidden_layer' + str(i-1)]))
                    sum = np.sum(hidden_error, axis=0)
                    parameters["b" + str(i)] += learning_rate * sum.reshape((3, 1))
                    print("Finish")

            elif (Activation == "Tanh"):
                output_error = errors * tanh_derivative(layers_outputs["output_layer"])
                GGG = output_error
                for i in range(len(layers_outputs) - 2, 0, -1):
                    hidden_error = np.dot(GGG, parameters["w" + str(i)]) * tanh_derivative(layers_outputs['hidden_layer' + str(i)])
                    GGG = hidden_error
                    parameters["w" + str(i)] += (learning_rate * np.dot(hidden_error.T, layers_outputs['hidden_layer' + str(i)]))
                    sum = np.sum(hidden_error, axis=0)
                    parameters["b" + str(i)] += learning_rate * sum.reshape(parameters["b" + str(i)].shape)

            return parameters

        def initialize_ps(layerdims):

            Layer = len(layerdims)
            for i in range(1, Layer):
                self.parameters['w' + str(i)] = np.random.randn(int(layerdims[i - 1]), int(layerdims[i]))
                if (self.Biasval == 0):
                    self.parameters['b' + str(i)] = np.zeros((layerdims[i], 1))
                else:
                    self.parameters['b' + str(i)] = np.full((layerdims[i], 1), self.Biasval)

        def PreProcessing():
            data = pd.read_csv("Dry_Beans_Dataset.csv")  # read file

            data = data.applymap(lambda x: str(x).replace("٫", "."))

            data = data.fillna(data.mean())

            Class1 = data.iloc[:50, :]  # seperate data and take first class
            Class2 = data.iloc[50:100, :]  # seperate data and take second class
            Class3 = data.iloc[100:, :]  # seperate data and take third class

            x_train1, x_test1, y_train1, y_test1 = train_test_split(Class1.iloc[:, :-1].values,
                                                                    Class1.iloc[:, -1].values, test_size=20,
                                                                    random_state=20)  # 30/20
            x_train2, x_test2, y_train2, y_test2 = train_test_split(Class2.iloc[:, :-1].values,
                                                                    Class2.iloc[:, -1].values, test_size=20,
                                                                    random_state=20)  # 30/20
            x_train3, x_test3, y_train3, y_test3 = train_test_split(Class3.iloc[:, :-1].values,
                                                                    Class3.iloc[:, -1].values, test_size=20,
                                                                    random_state=20)  # 30/20

            x_train1 = pd.DataFrame(x_train1, columns=Class1.iloc[:, :-1].columns)
            x_test1 = pd.DataFrame(x_test1, columns=Class1.iloc[:, :-1].columns)
            y_train1 = pd.DataFrame(y_train1, columns=[Class1.columns[-1]])
            y_test1 = pd.DataFrame(y_test1, columns=[Class1.columns[-1]])

            x_train2 = pd.DataFrame(x_train2, columns=Class2.iloc[:, :-1].columns)
            x_test2 = pd.DataFrame(x_test2, columns=Class2.iloc[:, :-1].columns)
            y_train2 = pd.DataFrame(y_train2, columns=[Class1.columns[-1]])
            y_test2 = pd.DataFrame(y_test2, columns=[Class1.columns[-1]])

            x_train3 = pd.DataFrame(x_train3, columns=Class3.iloc[:, :-1].columns)
            x_test3 = pd.DataFrame(x_test3, columns=Class3.iloc[:, :-1].columns)
            y_train3 = pd.DataFrame(y_train3, columns=[Class1.columns[-1]])
            y_test3 = pd.DataFrame(y_test3, columns=[Class1.columns[-1]])

            X_TRAIN1 = pd.concat([x_train1, x_train2, x_train3], axis=0)
            X_TRAIN = self.scaler.fit_transform(X_TRAIN1)
            X_TRAIN = pd.DataFrame(X_TRAIN, columns=X_TRAIN1.columns)

            X_TEST1 = pd.concat([x_test1, x_test2, x_test3], axis=0)
            X_TEST = self.scaler.fit_transform(X_TEST1)
            X_TEST = pd.DataFrame(X_TEST, columns=X_TEST1.columns)

            Y_TRAIN = pd.concat([y_train1, y_train2, y_train3], axis=0)
            Y_TRAIN['Class'] = self.label_encoder.fit_transform(Y_TRAIN)

            Y_TEST = pd.concat([y_test1, y_test2, y_test3], axis=0)
            Y_TEST['Class'] = self.label_encoder.fit_transform(Y_TEST)

            return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

            # region Main

        # region Main

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = PreProcessing()

        inp = [X_TRAIN.shape[1],]
        for i in self.neo_num_layer_values:
            inp.append(i)
        inp.append(3)
        initialize_ps(inp)

        #w1.shape = (5,3) first hidden
        #w2.shape = (3,3)  second hidden
        #w3.shape = (3,3)  output

        # b.shape = (3,1)

        #input = (90,5)


        for i in range(self.epochs):
            layers_outputs = forward_prop(X_TRAIN,self.parameters,self.actication)
            self.parameters = backward_prop(X_TRAIN,Y_TRAIN,layers_outputs,self.parameters,self.Learning,self.actication)

        # endregion

    def predict(self):
        print("prediction...")

