import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter.filedialog import askopenfilename, askdirectory
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import ImageTk

from detection import *
from training import train_meso4
from visualisation import *

# the Tkinter app class
class App:
    def __init__(self, app_root):
        # setting title
        app_root.title("Deepfake Detection App") # window title
        # setting window size
        width = 800
        height = 350
        screenwidth = app_root.winfo_screenwidth()
        screenheight = app_root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        app_root.geometry(alignstr)
        app_root.resizable(width=False, height=False)

        # the text labels
        titleLabel = tk.Label(app_root, text="Deepfake Detection Builder")
        titleLabel.place(x=0, y=0, width=401, height=37)

        trainingLabel = tk.Label(app_root, text="1. Training")
        trainingLabel.place(x=20, y=30, width=70, height=25)

        detectionLabel = tk.Label(app_root, text="2. Detection / Prediction", justify="left")
        detectionLabel.place(x=230, y=30, width=135, height=30)

        classiLabel = tk.Label(app_root, text="Metrics")
        classiLabel.place(x=20, y=260, width=70, height=30)

        # radio buttons
        self.epochs_var = tk.IntVar()
        radio_5Button = tk.Radiobutton(app_root, text="01 epochs", variable=self.epochs_var, value=1)
        radio_5Button.place(x=20, y=60, width=85, height=25)

        radio_10Button = tk.Radiobutton(app_root, text="5 epochs", variable=self.epochs_var, value=5)
        radio_10Button.place(x=20, y=85, width=85, height=25)

        radio_15Button = tk.Radiobutton(app_root, text="10 epochs", variable=self.epochs_var, value=10)
        radio_15Button.place(x=20, y=110, width=85, height=25)

        # radio buttons for model select for metrics
        self.saveResults = tk.BooleanVar()
        saveResults_Cb = tk.Checkbutton(app_root, text="Save Results", variable=self.saveResults, onvalue=True,
                                        offvalue=False)
        saveResults_Cb.place(x=270, y=263)

        self.model_option = tk.IntVar()
        radio_M4Button = tk.Radiobutton(app_root, text="Meso4", variable=self.model_option, value=0)
        radio_M4Button.place(x=110, y=263)

        # detection radio buttons
        self.test_option_var = tk.IntVar()
        random_testRButton = tk.Radiobutton(app_root, text="Random Test Data", variable=self.test_option_var, value=1)
        random_testRButton.place(x=230, y=60, width=120, height=25)

        radio_10Button = tk.Radiobutton(app_root, text="Single Image Test", variable=self.test_option_var, value=0)
        radio_10Button.place(x=230, y=85, width=120, height=25)

        # buttons
        self.meso4_trainingButton = tk.Button(app_root, text="Meso-4 Model", command=self.meso4_trainingLabel_command)
        self.meso4_trainingButton.place(x=20, y=160, width=120, height=40)

        self.meso4_detectionButton = tk.Button(app_root, text="Meso-4 Model", command=self.meso4_detectionLabel_command)
        self.meso4_detectionButton.place(x=230, y=160, width=120, height=40)

        self.classiButton = tk.Button(app_root, text="Classification Report\n& Confusion Matrix",
                                      command=self.classi_conf_report)
        self.classiButton.place(x=20, y=290, width=180, height=40)

        self.featureVisButton = tk.Button(app_root, text="Feature\nVisualisation", command=self.featureVis)
        self.featureVisButton.place(x=205, y=290, width=90, height=40)
        self.filterVisButton = tk.Button(app_root, text="Filter\nVisualisation", command=self.filterVis)
        self.filterVisButton.place(x=300, y=290, width=90, height=40)

        # check boxes - detection
        self.savefigVar = tk.BooleanVar()
        savefig_Cb = tk.Checkbutton(app_root, text="Save Figure", variable=self.savefigVar, onvalue=True,
                                    offvalue=False)
        savefig_Cb.place(x=230, y=110, height=25)

        self.showfigVar = tk.BooleanVar()
        showfig_Cb = tk.Checkbutton(app_root, text="Show Figure", variable=self.showfigVar, onvalue=True,
                                    offvalue=False)
        showfig_Cb.place(x=230, y=135, height=25)

        # image label - prediction
        frame = tk.Frame(app_root, background="black")
        frame.pack()
        frame.place(width=400, height=350, x=400, y=0)

        # image to display when detecting a single image
        self.imageLabel = tk.Label(frame, background="black")
        self.imageLabel.pack(anchor="center", pady=20)

        # label to set after a prediction is made
        self.pred_textvar = tk.StringVar()
        pred_label = tk.Label(app_root, textvariable=self.pred_textvar, background="black")
        pred_label.pack()
        pred_label.place(anchor="center", x=600, y=310)

        # image name to set after image is elected
        self.selImgVar = tk.StringVar()
        sel_imgLabel = tk.Label(app_root, textvariable=self.selImgVar, background="black")
        sel_imgLabel.pack()
        sel_imgLabel.place(anchor="w", x=400, y=5)

    # calls the meso4 detection function
    def meso4_detectionLabel_command(self):
        # getting the user variables from the checkboxes and radio buttons
        choice = self.test_option_var.get()
        saveFig = self.savefigVar.get()
        showFig = self.showfigVar.get()

        # if choice is singular image
        print(choice)
        if choice == 0:
            # catch exception if error occurs
            try:
                # ask for the image file
                path = askopenfilename(filetypes=[("Image Files", ['.jpg', '.png', '.bmp'])])

                # getting the image name
                imgname = path.split('/')[-1]
                actualLabel = path.split('/')[-2]

                # placing the image to the frame in the UI
                self.selImgVar.set(f"File: {imgname} - RESULT")
                image = ImageTk.PhotoImage(Image.open(path).resize((250, 250)))
                self.imageLabel.configure(image=image)
                self.imageLabel.image = image
                self.imageLabel.pack()

                # selecting what true label is from the file path
                if actualLabel == "fake":
                    actualLabel = "Deepfake"
                else:
                    actualLabel = "Real"

                # predictions from the model
                pred = meso4_det(choice, path)

                # displays the prediction on the User Interface
                if pred > 0.4:
                    self.pred_textvar.set(
                        f'Predicted Label: Deepfake\nLikelihood: {pred:.5f}\n Actual Label: {actualLabel}')
                elif pred > 0.25:
                    self.pred_textvar.set(
                        f'Predicted Label: Most likely a deepfake\nLikelihood: {pred:.5f}\n Actual Label: {actualLabel}')
                else:
                    self.pred_textvar.set(
                        f'Predicted Label: Real Image\nLikelihood: {pred:.5f}\n Actual Label: {actualLabel}')

            except Exception as e:
                print(f"\nError: {e}\n:Please select an image")

        # if the choice is random images from directory
        elif choice == 1:
            # catch exception
            try:
                # ask the user for directory
                dir_path = askdirectory()

                print(dir_path)

                # get the sample size from the  user
                sample_size = simpledialog.askinteger(title="Enter the sample size of images",
                                                          prompt="Enter sample size: ")
                print(dir_path)

                # call the function with the appropriate arguments
                meso4_det(choice, None, dir_path, sample_size, saveFig, showFig)

            except Exception as e:
                    print(f"\nError: {e}")

    # starts the training for the meso4 model
    def meso4_trainingLabel_command(self):
        epochs = self.epochs_var.get()
        print("---------------------------------------")
        print("::: Meso4 - Starting training...")
        print(f"::: Epoch is set to {epochs}")
        print("---------------------------------------")
        train_meso4(epochs)
        messagebox.showinfo("Message", f"Training for Meso4 has been completed with {epochs} epochs!")


    # generate confusion matrix and classification report
    def classi_conf_report(self):
        self.classiButton['state'] = 'disabled'
        model_selected = ["Meso4"]
        model_sel = self.model_option.get()
        save = self.saveResults.get()
        dir_path = askdirectory()

        if dir_path:
            print("---------------------------------------")
            print("::: Classification Report and Confusion Matrix")
            print(f"::: Model being used is {model_selected[model_sel]}")
            print(f"::: Dataset path is {dir_path}")
            print("---------------------------------------")

            gen_confi_conf(dir_path, save)

        self.classiButton['state'] = 'active'

    # generate feature maps
    def featureVis(self):
        model_selected = ["Meso4"]
        model_sel = self.model_option.get()
        save = self.saveResults.get()

        try:
            img_path = askopenfilename(filetypes=[("Image Files", '.jpg')])

        except Exception as e:
            print("Error: no image selected ", e)

        if img_path:
            print("---------------------------------------")
            print(f"::: Selected {model_selected[model_sel]} - generating feature visualisation...")
            print("---------------------------------------")
            feature_vis(model_sel, img_path, save)

    # generate filter plots
    def filterVis(self):
        model_selected = ["Meso4"]
        model_sel = self.model_option.get()
        save = self.saveResults.get()

        print("---------------------------------------")
        print(f"::: Selected {model_selected[model_sel]} - generating feature visualisation...")
        print("---------------------------------------")
        filter_vis(model_sel, save)

# start the app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

