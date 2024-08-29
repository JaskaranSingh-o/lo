# imports
import glob
import itertools
from datetime import date, datetime

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from classifier import Meso4
from utils import *


# dont display tensorflow warning in the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# gets the images from the directories
def predicts(model, img_w, img_h, dir_path):
    # obtains all the images in a folder
    real = glob.glob(rf'{dir_path}/real/*.*[png|jpg]*')
    df = glob.glob(rf'{dir_path}/fake/*.*[png|jpg]*')

    labels = []
    preds = []

    # predict on images and add the predicting in an array along with the image label
    print("Predicting on all the images in real label")
    for i in tqdm(range(len(real))):
        img = image_to_array(real[i], img_w, img_h)
        X = model.predict(img)
        preds.append(round(X[0][0]))
        labels.append(1)

    # predict on images and add the predicting in an array along with the image label
    print("Predicting on all the images in deepfake label")
    for i in tqdm(range(len(df))):
        img = image_to_array(df[i], img_w, img_h)
        X = model.predict(img)
        preds.append(round(X[0][0]))
        labels.append(0)

    # experimental
    # r_l = list(itertools.repeat(1, len(real)))
    # f_l = list(itertools.repeat(0, len(df)))
    # labels = r_l + f_l

    # return labels and prediction arrays
    return labels, preds


# generate confusion matrix
def confusion_mat(model_sel=1, dir_path='real-vs-fake/test', save=False):
    # catch errors
    try:
        # if model is meso4 set these variables
        if model_sel == 0:
            model = Meso4()
            # model.load('weights/Meso4_Custom.h5')
            model.load('weights/Meso4_Custom.h5')
            img_w, img_h = 256, 256

        # call the predicts() function with model selected, input img width and heigh and directory path for images
        labels, preds = predicts(model, img_w, img_h, dir_path)

        # generate the confusion matrix using the labels and predictions from the predicts() function
        cm = confusion_matrix(labels, preds)
        # create a figure to plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        # create a plot from the confusion matrix generated
        hm_plot = sns.heatmap(cm, annot=True, fmt='.0f')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M")
        # plot the matrix on the plot
        fig = hm_plot.get_figure()

        # if save option is selected, save the plot
        if save:
            fig.savefig(f'Metrics/confusion_matrix-{"Meso4" if model_sel == 0 else "AlexNet"}.png')
        plt.show()
    except Exception as e:
        print(e)


# generate classification report
def class_report(model_sel=1, dir_path='real-vs-fake/test', save=False):
    # if model is meso4 set these variables
    if model_sel == 0:
        model = Meso4()
        model.load('weights/Meso4_Custom.h5')
        img_w, img_h = 256, 256

  # call the predicts() function with model selected, input img width and heigh and directory path for images
    labels, preds = predicts(model, img_w, img_h, dir_path)

    # generate the classification report

    c_report = classification_report(labels, preds)
    print('Classification Report')
    print(c_report)

    # if save is selected, save the report to a text file
    if save:
        classi_to_file(c_report, 'Metrics/')


# generate both confusion matrix and classification report.
def gen_confi_conf(dir_path='real-vs-fake/test', save=False):
    # initiate the models and their weights
    modelm4 = Meso4()
    modelm4.load('weights/Meso4_Custom.h5')

    # get the labels and predictions for both models
    labels_m4, preds_m4 = predicts(modelm4, 256, 256, dir_path)

    # generate classification report
    c_report_M4 = classification_report(labels_m4, preds_m4)

    # print both classification report
    print('Classification Report - Meso4')
    print(c_report_M4)
    print("\n")



    # catch errors - create and plot confusion matrix
    try:
        cm = confusion_matrix(labels_m4, preds_m4)
        fig, ax = plt.subplots(figsize=(10, 10))
        hm_plot = sns.heatmap(cm, annot=True, fmt='.0f')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M")
        fig = hm_plot.get_figure()
        if save:
            classi_to_file(c_report_M4, 'Metrics/', "Meso4")
            fig.savefig(f'Metrics/confusion_matrix-Meso4.png')
        plt.show()
    except Exception as e:
        print(e)

# feature map visualization
def visu_feature_filter(model_sel, data, save):
    sel_mode = ""
    square = 8
    fea_i = 0
    # check if save directory exist, if not, create
    check_create_dir('FeatureDirectory/')

    # generate the feature map plots
    # iterate over how many layers there are
    for index, fmap in enumerate(data):
        # get which model is selected and the number of cols and rows to display
        ix = 1
        if index == 0 and model_sel == 0 or index == 1 and model_sel == 0:
            sel_mode = "Meso4"
            columns = 2
            rows = 4
        elif model_sel == 0 and index > 1:
            sel_mode = "Meso4"
            columns = 4
            rows = 4

# create a new figure
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"Deepfake Detection - {sel_mode} Feature Visualization - Layer {fea_i}")

        # plotting the feature maps in the figure; col * rows
        for _ in range(1, columns * rows + 1):
            # specify subplot and turn of axis
            # create a subplot
            ax = plt.subplot(columns, rows, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            try:
                plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
            except Exception as e:
                print(e)
            ix += 1
        # show and/or save the figure
        if save:
            plt.savefig(f"FeatureDirectory/{sel_mode}_layer_{fea_i}-{columns}x{rows}_figure.png")
        plt.show()
        fea_i += 1

def feature_vis(model_sel=1, img=None, save=False):
    # if model is meso4 set these variables
    if model_sel == 0:
        model = Meso4()
        model.load('weights/Meso4_Custom.h5')
        img_w, img_h = 256, 256

# get the convolutional layers from the classifiers
    conv_layers, _ = model.get_conv_layers()
    img = image_to_array(img, img_w, img_h) # image to array function

    # get the filters from the classifiers
    model = model.filters(conv_layers)

    # get the feature maps by predicting on the image
    feature_maps = model.predict(img)

    # call the function to plot the maps on a plot
    visu_feature_filter(model_sel, feature_maps, save)


# plot the filters on the graph
def filter_vis(model_sel=1, save=False):
    # if model is meso4 set these variables
    sel_mode = ""
    if model_sel == 0:
        model = Meso4()
        model.load('weights/Meso4_Custom.h5')
        sel_mode = "Meso4"

    # check if a directory exists, if not, create
    check_create_dir('FilterDirectory/')

    # get the convolutional layers and the layer number
    conv_layers, layer_ = model.get_conv_layers()

    # iterate over layer data (filters)
    for index, layer in enumerate(layer_):
        # get which model is selected and the number of cols and rows to display
        if index == 0 and model_sel == 0 or index == 1 and model_sel == 0:
            sel_mode = "Meso4"
            columns = 2
            rows = 4
        elif model_sel == 0 and index > 1:
            sel_mode = "Meso4"
            columns = 4
            rows = 4

            # get the filter information to plot them
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)

            # normalize filter values between  0 and 1 for visualization
            fmin, fmax = filters.min(), filters.max()
            filters = (filters - fmin) / (fmax - fmin)

            # 6 filters to dispaly
            n_filters = 6
            ix = 1
            fig = plt.figure()  # create figure
            # 6 filters to display
            for i in range(n_filters):
                # get the filters
                f = filters[:, :, :, i]
                for j in range(3):
                    # subplot for 6 filters and 3 channels
                    ax = plt.subplot(n_filters, 3, ix)
                    # plot the filters in grayscale
                    plt.imshow(f[:, :, j], cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ix += 1

            # show and/or save the figure
            if save:
                plt.savefig(f"FilterDirectory/{sel_mode}_layer_{layer.name}-{n_filters}x{3}_figure.png")

            plt.show()
