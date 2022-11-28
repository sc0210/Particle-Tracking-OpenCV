import glob
import os
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import signal
from scipy.signal import fftconvolve
from tqdm import tqdm, trange
from time import sleep

def EnvSetup(path):
    if os.path.exists(path) == False:
        os.mkdir(path)


def MakeSubFolders(root_dir, subfolders):
    # for index in ['ROI','GIF','Plot','Reserved']:
    #   subfolders.append(index)
    concat_path = partial(os.path.join, root_dir)
    for subfolder in map(concat_path, subfolders):
        os.makedirs(subfolder, exist_ok=True)  # Python 3.2+


def ReadGrayImg(SrcPath, show=False):
    """
    Read an gray image and option either to show on the window.

    Args:
        SrcPath:(str) image path
        show:(bool) show the image on the window
    """
    Img = cv2.imread(SrcPath, cv2.IMREAD_GRAYSCALE)
    if show == True:
        cv2.imshow('image', Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return Img


def IMG2MP4(SrcFolder, OutFolder="./Export", OutName="test", FPS=5):
    """
    Convert sequential TIFs images into AVI clip format.

    Args:
        SrcFolder: (str) input folder locaiton (picture file)
        OutFolder: (str) output folder location (video file)
        OutName: (str) AVI clip name
        FPS: (int) frame per seonds
    """
    print(f"Converting to MP4...{OutName}")
    FPS = int(FPS)  # Frame per seconds
    SrcFolder = str(SrcFolder)
    OutFolder = str(OutFolder)
    video_name = f'{OutFolder}/{OutName}.mp4'

    # Read every .Tif files in the folder

    ImgType = os.path.splitext(os.listdir(SrcFolder)[0])[1]
    if ImgType == ".tif":
        images = [img for img in os.listdir(SrcFolder) if img.endswith(".tif")]
    elif ImgType == ".png":
        images = [img for img in sorted(
            os.listdir(SrcFolder)) if img.endswith(".png")]
    elif ImgType == ".jpg":
        images = [img for img in sorted(
            os.listdir(SrcFolder)) if img.endswith(".jpg")]
    else:
        print("Not recongized image type")

    # Fetch the frame shape(e.g height, width, layer)
    frame = cv2.imread(os.path.join(SrcFolder, images[0]))
    height, width, layers = frame.shape

    # Adjust output location
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, FPS, (width, height))

    # Convert images(.tif) into video(.avi)
    for image in images:
        video.write(cv2.imread(os.path.join(SrcFolder, image)))

    cv2.destroyAllWindows()
    video.release()


def PNG2GIF(SrcFolder, OutFolder, OutName, ImgFormat="png", duration=120):
    """
    Convert sequential PNG images into GIF clip format.

    Args:
        SrcFolder: (str) input folder locaiton (picture file)
        OutFolder: (str) output folder location (video file)
        OutName: (str) AVI clip name
        ImgFormat:(str, default="png") image's type to be transform
        duration:(int) GIF clips long
    """
    frames = [Image.open(image) for image in sorted(
        glob.glob(f"{SrcFolder}/*.{ImgFormat}"))]
    frame_one = frames[0]
    frame_one.save(f"{OutFolder}/GIF/{OutName}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=0)
    print(f"Converting to GIF...{OutName}")


def normxcorr2(template, image, mode="full"):
    ########################################################################################
    # Author: Ujash Joshi, University of Toronto, 2017                                     #
    # Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
    # Octave/Matlab normxcorr2 implementation in python 3.5                                #
    # Details:                                                                             #
    # Normalized cross-correlation. Similiar results upto 3 significant digits.            #
    # https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
    # http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
    # """
    """
    Input arrays should be floating point numbers.

    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.

    :param image: N-D array

    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
        np.square(fftconvolve(image, a1, mode=mode)) / \
        (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0
    #template[np.where(template < 0)] = 0
    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


def Track(SrcFolder, DstFolder, GroupIndex):
    """
    Track the circle particle in the pictures.
    Return the track point result in format of two list (x_list, y_list)

    Args:
        SrcFolder: (str) input folder locaiton (picture file)
        OutFolder: (str) output folder location (video file)
        OutName: (str) AVI clip name
        SavePlot:(bool) whether to save the reuslts (sequiential figures)
    """
    Corr = [0, 0, 0, 0, 0, 0, 0]  # x1, x2, y1, y2, centerx, centery, radius
    list_x, list_y = [], []

    # ---------ROI section--------------------------------------------------------------------------------
    ROI_INDEX = np.random.randint(0, len(os.listdir(f"{SrcFolder}/{GroupIndex}")))

    tempPath = f"{SrcFolder}/{GroupIndex}"
    template = cv2.imread(
        f"{SrcFolder}/{GroupIndex}/{sorted (os.listdir(tempPath))[ROI_INDEX]}",0)
    output = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(template, (5, 5), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                               1, 20, param1=50, param2=20,
                               minRadius=5, maxRadius=25)
    print("////"*18)
    print(f"=> Currenet analysis image: {GroupIndex}")
    if circles is not None: # Get the (x, y, r) as integers
        circles = np.uint16(np.around(circles))
        # loop over the circles
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2],(73, 235, 52), 1)  # draw outer
            #cv2.circle(image, center_coordinates, radius, color, thickness)
            # Thickness of -1 px will fill the circle shape by the specified color.
            cv2.circle(output, (i[0], i[1]), 1, (255, 0, 0), 1)  # draw center
            interval = 3
            x1 = i[0] - (i[2]+interval)
            x2 = i[0] + (i[2]+interval)
            y1 = i[1] - (i[2]+interval)
            y2 = i[1] + (i[2]+interval)
            Corr[1], Corr[0], Corr[2], Corr[3] = y1, x1, x2, y2
            Corr[4] = i[0]  # center X
            Corr[5] = i[1]  # center Y
            Corr[6] = i[2]  # circle radius
            output = output[Corr[1]:Corr[3], Corr[0]:Corr[2]]
            cv2.imwrite(f"./Export/TrackFile/ROI/ROI_{GroupIndex}.png", output)

            print(
                f"CenterX,CenterY:({i[0]},{i[1]}); radius:{i[2]}; (x1,y1):({x1},{y1}),(x2,y2):({x2},{y2})")
    else:
        print("There is no circles in the picture! Please check~\n")

    roi_initial = template.copy()
    roi = cv2.GaussianBlur(roi_initial,(5,5), 0)
    roi = roi[Corr[1]:Corr[3], Corr[0]:Corr[2]]
    progress =tqdm(total=len(os.listdir(f"{SrcFolder}/{GroupIndex}")))
    times = 0

    # ---------NormalXCorr--------------------------------------------------------------------------------
    nfile=[]
    files = sorted(os.listdir(f"{SrcFolder}/{GroupIndex}"))
    for x in files:
        if x.split(".")[1] == "tif":
           nfile.append(x)

    for filename in sorted(nfile):
        image_initial = cv2.imread(f"{SrcFolder}/{GroupIndex}/{filename}", 0)
        image = cv2.GaussianBlur(image_initial,(5,5), 3)

        # Cross correlation
        corr = normxcorr2(roi, image, mode="same")
        y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
        list_x.append(x); list_y.append(y)
        progress.update(1)
        times += 1
        sleep(0.01)

    # ---------Save2Excel--------------------------------------------------------------------------------
    data = pd.DataFrame({'x_axis': list_x, 'Y_axis': list_y})
    data.to_excel(f"{DstFolder}/Result_{GroupIndex}.xlsx",
                    sheet_name='XY', index=False)

    # ---------Plot Figure--------------------------------------------------------------------------------
    """ Some
    
    
    for filename in sorted(nfile):
        image_initial = np.asarray(cv2.imread(f"{SrcFolder}/{GroupIndex}/{filename}", 0))
        fig, (ax_orig, ax_corr) = plt.subplots(1, 2)
        ax_orig.imshow(image_initial, cmap="gray")
        ax_orig.set_title(f'{filename} (Image)')
        ax_orig.set_axis_off()

        index = sorted(os.listdir(f"{SrcFolder}/{GroupIndex}")).index(filename)
        # Plot xy-center
        ax_orig.plot(list_x[index], list_y[index], 'ro', markersize=4)
        # Plot whole track
        #ax_orig.plot(list_x[index-5:index], list_y[index-5:index],'g',marker='.',linewidth="1",markersize=5,alpha=0.8)

        # Plot ROI reference
        ax_corr.imshow(output, cmap="gray")
        ax_corr.set_title(f'No.{ROI_INDEX} (ROI/Template)')
        ax_corr.set_axis_off()

        filename = filename.replace(".tif", "")
        plt.savefig(f"{DstFolder}/{GroupIndex}/{filename}.png",bbox_inches='tight')
        plt.cla()
        plt.close(fig)
    """
    print(f"image shape:{image_initial.shape}, ROI:{roi.shape}\n")
    return list_x, list_y


def MSD(X, Y, FPS, DstFolder, GroupIndex, ImgShow=False):
    """
    Plot Mean Square Displacement(MSD) figure.

    Args:
        X:(array) 1-D array of track result
        Y:(array) 1-D array of track result
        OutFolder: (str) output folder location
        GroupIndex: (str) figure name
        ImgShow: (bool, default="False") Whether the show the figure
    """
    print(f"=> Calculating MSD...")
    sol = []
    length = len(X)
    count, LAG = 0, 5

    for interval in range(1, length, LAG):  # Loop interval
        dx1, dy1 = [], []
        avg_x, avg_y = 0, 0
        for i in range(0, length):  # Loop in single string
            if (i+interval) < length:
                dx1.append(float(X[i+interval] - X[i]) ** 2)
                dy1.append(float(Y[i+interval] - Y[i]) ** 2)
        count += 1
        #print(dx1,dy1)
        avg_x, avg_y = float(sum(dx1)/len(dx1)), float(sum(dy1)/len(dy1))
        sol.append(avg_x + avg_y)
        # print(avg_x,avg_y)

    sns.set_theme(color_codes=True)
    fig = plt.figure(figsize=(8, 5))
    fig = sns.regplot(x=np.linspace(0, (count-1)/FPS, count),
        y=sol, marker='o', label="example", order=1,
        robust=False, ci=None,     
        scatter_kws={'s': 10, 'color': '#7d46b4'},
        line_kws={'lw': 2, 'color': '#b4466e'})
    # Seaborn Resources
    # https://seaborn.pydata.org/generated/seaborn.regplot.html
    # Robust: bool, will de-weight outliers
    fig.figure.autofmt_xdate()
    plt.title(f'Linear Regression of: {GroupIndex}', fontsize=15)
    plt.xlabel("t(sec)", fontsize=15)
    plt.ylabel("MSD(square of meters)", fontsize=15)
    plt.savefig(f"{DstFolder}/Plot/{GroupIndex}.png")

    # plt.tight_layout()
    if ImgShow == True:
        plt.show()
    
    return sol


def GetCenter(Image_path, OutName):
    """
    Track single frame based on HoughCircles algorithm from OpenCV.

    Args:
        Image_path:(str) image path
        OutName:(str) name for the output filename
    """
    Corr = [0, 0, 0, 0, 0, 0, 0]  # (x1, x2, y1, y2, centerx, centery, radius)
    Result = [0, 0]  # get (X,Y) from HoughCircles
    
    print("////"*18)
    print(f"=> Currenet analysis image: {OutName}")

    template = cv2.imread(Image_path,0) # Read Gray Image
    output = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(template,cv2.HOUGH_GRADIENT,1,20, 
                                param1=50, param2=20, minRadius=5, maxRadius=16)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2],
                       (73, 235, 52), 1)  # draw outer
            #cv2.circle(image, center_coordinates, radius, color, thickness)
            # Thickness of -1 px will fill the circle shape by the specified color.
            cv2.circle(output, (i[0], i[1]), 1, (255, 0, 0), 1)  # draw center
            interval = 5
            x1 = i[0] - (i[2]+interval)
            x2 = i[0] + (i[2]+interval)
            y1 = i[1] - (i[2]+interval)
            y2 = i[1] + (i[2]+interval)
            Corr[1], Corr[0], Corr[2], Corr[3] = y1, x1, x2, y2
            Corr[4], Corr[5], Corr[6] = i[0], i[1], i[2]  # center X, center Y, circle radius
            output = output[Corr[1]:Corr[3], Corr[0]:Corr[2]]
            cv2.imwrite(f"./Export/TrackFile/ROI/ROI_{OutName}.png", output)

            print(f"=> Results: CenterX,CenterY: ({i[0]},{i[1]}) | radius:{i[2]}\n")
            Result[0], Result[1] = i[0], i[1]
            return Result
    else:
        print("There is no circles in the picture! Please check~\n")


def Threshold(image, algo=cv2.THRESH_OTSU):
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    ret, threshold_image = cv2.threshold(blur, 0, 255, algo)
    # dst = cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
    return threshold_image


def SeqNCC(TrackX, TrackY, idx, root, DstFolder, GroupIndex, image_name):
    """
    Track sequence of the image and save in TrackX, TrackY

    Args:
        TrackX(str):
        TrackY(str):
        idx(int): 
        root(str): 
        DstFolder(str): 
        GroupIndex(int):
        image_name(str):
    """
    nfile=[]
    files = sorted(os.listdir(root))
    for x in files:
        if x.split(".")[1] == "tif":
           nfile.append(x)

    # --------Current analysis image index------------------------------------------------------------
    image_name = image_name.replace(".tif", "")
    roi_image = nfile[idx]
    roi_initial = np.asarray(cv2.imread(f"{root}/{roi_image}", 0))
    image_initial = np.asarray(cv2.imread(f"{root}/{image_name}.tif", 0))

    BBOX_size = 14  # Including radius and padding
    x1 = TrackX[idx] - BBOX_size
    x2 = TrackX[idx] + BBOX_size
    y1 = TrackY[idx] - BBOX_size
    y2 = TrackY[idx] + BBOX_size
    #print(f"x1:{x1}, x2:{x2}, y1:{y1}, y2:{y2}")

    # --------Cross correlation------------------------------------------------------------------------
    roi = roi_initial[y1:y2, x1:x2]
    th3 = Threshold(image_initial, algo=cv2.THRESH_OTSU)
    th2 = Threshold(roi, algo=cv2.THRESH_OTSU)

    corr = normxcorr2(roi, image_initial, mode="same")
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

    # Plot the Track point(X,Y) overlays with original pictures
    fig, (ax_orig, ax_corr) = plt.subplots(1, 2)
    ax_orig.imshow(image_initial, cmap="gray")
    ax_orig.set_title(f'(Image){image_name} ')
    ax_orig.set_axis_off()

    # --------Plot xy-center---------------------------------------------------------------------------
    ax_orig.plot(x, y, 'ro', markersize=6)
    # Plot whole track
    ax_orig.plot(TrackX[idx-5:idx+2], TrackY[idx-5:idx+2],
                 'g', marker='.', linewidth="1", markersize=5, alpha=0.8)

    # --------Plot ROI reference------------------------------------------------------------------------
    ax_corr.imshow(roi, cmap="gray")
    ax_corr.set_title(f"(ROI/Template){image_name} ")
    ax_corr.set_axis_off()

    plt.savefig(f"{DstFolder}/{GroupIndex}/{image_name}.png",
                bbox_inches='tight')
    plt.cla()
    plt.close(fig)
    return x, y


def Track2(SrcFolder, DstFolder, GroupIndex):
    root = f"{SrcFolder}/{GroupIndex}"
    TrackX, TrackY = [], []
    progress = tqdm(total=len(os.listdir(root)))
    times = 0

    nfile=[]
    files = sorted(os.listdir(root))
    for x in files:
        if x.split(".")[1] == "tif":
           nfile.append(x)

    first_image_name = nfile[0]
    # --------Return X,Y from first picture----------------------------------------------------------------
    TrackResult = GetCenter(f"{root}/{''.join(first_image_name)}", GroupIndex)
    TrackX.append(TrackResult[0]); TrackY.append(TrackResult[1])
    ### print(f"=> TrackX:{TrackX}, TrackY:{TrackY}")

    # --------Rest sequential image will track by the former image----------------------------------------
    for image in sorted(os.listdir(root))[1:]:
        # Use previous image as Cross-Correlation
        prev_idx = sorted(os.listdir(root)).index(image) - 1

        if prev_idx < len(os.listdir(root)):
            # idx stands for the former one (ROI)
            x, y = SeqNCC(TrackX, TrackY, prev_idx, root, DstFolder, GroupIndex, image)
            #print(f"x:{x}, y:{y}")
            TrackX.append(x)
            TrackY.append(y)
            progress.update(1); times += 1; sleep(0.01)

    # ---------Save2Excel--------------------------------------------------------------------------------
    data = pd.DataFrame({'x_axis': TrackX, 'Y_axis': TrackY})
    data.to_excel(f"{DstFolder}/Result_{GroupIndex}.xlsx",
                  sheet_name='XY', index=False)

    return TrackX, TrackY
