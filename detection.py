import glob
from datetime import datetime

from utils import *


def meso4_det(choice, image_path=None, dir_path=None, sample=9, saveFig=False, showFig=True):
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    # 1 - Load the model and its pretrained weights
    classifier_m4 = Meso4()
    classifier_m4.load('weights/Meso4_Custom.h5')

    # 0 == individual image
    if choice == 0:
        # convert image to array
        img = image_to_array(image_path, 256, 256)

        # make a prediction
        pred = classifier_m4.predict(img)

        # print the prediction in console
        pred = 1 - pred[0][0]
        print(f"Pred: {pred}")
        if pred > 0.4:
            print("Deepfake")
            print(f"Likelihood: {pred:.5f}")
        elif pred > 0.25:
            print("Most likely a deepfake")
            print(f"Likelihood: {pred:.5f}")
        else:
            print("Real Image")
            print(f"Likelihood: {pred:.5f}")
        print("\n")
        return pred # return the prediction to the user interface


    # choice 1 == multi-image detection
    if choice == 1:

        img_list = []
        data = []

        try:
            # get the director content
            dir_content = glob.glob(f"{dir_path}/*.*[png|jpg]*")[:sample]

            # iterate over images in the list and get their name
            for img in dir_content:
                img_list.append(img.split('\\')[-1])

        except Exception as e:
            print(f"Error: {e}")

        # make prediction on the images and add them to a dictionary
        for i in img_list:
            img_path = os.path.join(dir_path, i)
            img = image_to_array(img_path, 256, 256)

            X = classifier_m4.predict(img)
            data_dict = {i: X}
            data.append(data_dict)

        # plot the images on a graph
        fig = plt.figure(figsize=(16, 9))
        fig.canvas.manager.set_window_title("Deepfake Detection using Meso4 Model")
        fig.suptitle(
            "Deepfake Detection.\nLikelihood scale: being close to 1 is considered a deepfake")
        counter = 0
        for _ in data:
            for key, val in _.items():
                fig.add_subplot(3, 4, counter + 1)
                imgToShow = image_to_array(os.path.join(dir_path, key), 256, 256)
                plt.imshow(np.squeeze(imgToShow))
                plt.subplots_adjust(hspace=0.5)

                pred = 1 - val[0][0]
                if pred > 0.4:
                    plt.title("Deepfake")
                    plt.xlabel(f"Likelihood: {pred:.5f}")
                elif pred > 0.25:
                    plt.title("Most likely a deepfake")
                    plt.xlabel(f"Likelihood: {pred:.5f}")
                else:
                    plt.title("Real Image")
                    plt.xlabel(f"Likelihood: {pred:.5f}")
                plt.ylabel(f"{key}")
                ax = plt.gca()
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                counter += 1

        # if user selected save - save the figure
        if saveFig:
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y %H-%M-%S")
            plt.savefig(f"Figure-{date_time}")

        # if user selected show - show the figure
        if showFig:
            plt.show()

