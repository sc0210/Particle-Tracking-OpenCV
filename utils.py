import cv2, os, glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import fftconvolve
from functools import partial
import seaborn as sns

def EnvSetup(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

def MakeSubFolders(root_dir, subfolders):
    for index in ['ROI','GIF','Plot']:
        subfolders.append(index)
    concat_path = partial(os.path.join, root_dir)
    for subfolder in map(concat_path, subfolders):
        os.makedirs(subfolder, exist_ok=True)  # Python 3.2+

def ReadGrayImg(SrcPath, show=False):
    """
    Read an image.
        
    Args:
        SrcPath:(str)image path
        show:(bool)show the image on the screen
    """
    Img = cv2.imread(SrcPath,0)
    if show ==True:
        Img.show()
    return Img

def IMG2MP4(SrcFolder,OutFolder="./Export",OutName="test",FPS=5):
    """
    Convert sequential TIFs images into AVI clip format.
        
    Args:
        SrcFolder: (str) input folder locaiton (picture file)
        OutFolder: (str) output folder location (video file)
        OutName: (str) AVI clip name
        FPS: (int) frame per seonds
    """
    print(f"Converting to MP4...{OutName}")
    FPS = int(FPS) # Frame per seconds
    SrcFolder = str(SrcFolder); OutFolder = str(OutFolder)
    video_name = f'{OutFolder}/{OutName}.mp4'
            
    # Read every .Tif files in the folder

    ImgType = os.path.splitext(os.listdir(SrcFolder)[0])[1]
    if ImgType == ".tif":
        images = [img for img in os.listdir(SrcFolder) if img.endswith(".tif")]
    elif ImgType == ".png":
        images = [img for img in sorted(os.listdir(SrcFolder)) if img.endswith(".png")]
    else:
        print("Not recongized image type")
       
    # Fetch the frame shape(e.g height, width, layer) 
    frame = cv2.imread(os.path.join(SrcFolder, images[0]))
    height, width, layers = frame.shape
            
    # Adjust output location
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, FPS, (width,height))
            
    # Convert images(.tif) into video(.avi)
    for image in images:
        video.write(cv2.imread(os.path.join(SrcFolder, image)))
            
    cv2.destroyAllWindows()
    video.release()

def PNG2GIF(SrcFolder,OutFolder ,OutName,ImgFormat="png", duration=120):
    frames = []
    frames = [Image.open(image) for image in sorted(glob.glob(f"{SrcFolder}/*.{ImgFormat}"))]
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
    ########################################################################################"""
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
        np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0
    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
    img1 = cv2.GaussianBlur(img,size,sigma)
    img2 = cv2.GaussianBlur(img,size,sigma*k)
    return (img1-gamma*img2)
        
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
    aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
    for i in range(0,aux.shape[0]):
        for j in range(0,aux.shape[1]):
            if(aux[i,j] < epsilon):
                aux[i,j] = 1*255
            else:
                aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
    return aux

def xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
    aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
    for i in range(0,aux.shape[0]):
        for j in range(0,aux.shape[1]):
            if(aux[i,j] >= epsilon):
                aux[i,j] = 1
            else:
                ht = np.tanh(phi*(aux[i][j] - epsilon))
                aux[i][j] = 1 + ht
    return aux*255

def Track(SrcFolder, OutFolder,OutName="test" ,SavePlot=True):
    Corr =[0,0,0,0,0,0,0] #x1, x2, y1, y2, centerx, centery, radius
    list_x ,list_y = [],[]

    #---------ROI section--------------------------------------------------------------------------------
    ROI_INDEX = np.random.randint(0, len(os.listdir(SrcFolder)))

    template = ReadGrayImg(f"{SrcFolder}/{sorted(os.listdir(SrcFolder))[ROI_INDEX]}", False)        
    output = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    blur = cv2.medianBlur(template,3)  
    #blur = cv2.GaussianBlur(template,(7,7),0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,1,20,param1=50,param2=20,minRadius=5,maxRadius=25)
    print("////"*18)
    print(f"Currenet analysis image: {OutName}")
    if circles is not None:
        # Get the (x, y, r) as integers
        circles = np.uint16(np.around(circles))
        #print(circles)
        # loop over the circles
        for i in circles[0,:]:
            cv2.circle(output,(i[0],i[1]),i[2],(255, 100, 227),1) # draw outer
            #cv2.circle(image, center_coordinates, radius, color, thickness)
            #Thickness of -1 px will fill the circle shape by the specified color.
            cv2.circle(output,(i[0],i[1]),1,(255,0,0),1) # draw center
            interval = 3
            x1= i[0] - (i[2]+interval); Corr[0]=x1
            x2= i[0] + (i[2]+interval); Corr[2]=x2
            y1= i[1] - (i[2]+interval); Corr[1]=y1
            y2= i[1] + (i[2]+interval); Corr[3]=y2
            Corr[4]=i[0] # center X
            Corr[5]=i[1] # center Y
            Corr[5]=i[2] # circle radius
            output = output[Corr[1]:Corr[3], Corr[0]:Corr[2]]
            cv2.imwrite(f"./Export/TrackFile/ROI/ROI_{OutName}.png",output)

            print(f"CenterX,CenterY:({i[0]},{i[1]}); radius:{i[2]}; x1,y1:({x1},{y1}), (x2,y2):({x2},{y2})")
    else:
        print("There is no circles in the picture! Please check~\n")  
        
    
    roi_initial = np.asarray(template.copy())
    roi = cv2.medianBlur(roi_initial,5)
    #roi = xdog_garygrossi(roi,sigma=0.5,k=100, gamma=0.98,epsilon=0.1,phi=10)
    roi = roi[Corr[1]:Corr[3], Corr[0]:Corr[2]] #26:44,18:34]# opposite input seleciton

    #---------NormalXCorr--------------------------------------------------------------------------------
    for filename in sorted(os.listdir(SrcFolder)):
        #num1, num2 = str(index).zfill(4), str(index+1).zfill(4)
        image_initial = np.asarray(ReadGrayImg(f"{SrcFolder}/{filename}", False))
        image = cv2.medianBlur(image_initial,5)
        image = xdog_garygrossi(image,sigma=0.5,k=100, gamma=0.98,epsilon=0.1,phi=10)
                   
        ## Cross correlation
        corr = normxcorr2(roi, image, mode="same")
        y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
        list_x.append(x),list_y.append(y)
        
        if SavePlot == True:
            fig, (ax_orig, ax_corr) = plt.subplots(1, 2)
            ax_orig.imshow(image_initial, cmap='gray')
            ax_orig.set_title(f'{filename} (Image)')
            ax_orig.plot(x, y, 'ro',linewidth=2, markersize=12)
            ax_orig.set_axis_off()

            ax_corr.imshow(output, cmap='gray')
            ax_corr.set_title(f'No.{ROI_INDEX} (ROI/Template)')
            ax_corr.set_axis_off()
            fig.tight_layout()
            filename = filename.replace(".tif","")
            plt.savefig(f"{OutFolder}/{OutName}/{filename}.png", bbox_inches='tight')
            plt.cla()
            plt.close(fig)         
    print(f"image shape:{image_initial.shape}, ROI:{roi.shape}\n")
    return list_x, list_y

def MSD(X ,Y,OutFolder,filename,ImgShow=False):
    sol=[];y=[]; length=len(X)
    for interval in range(1,length): # Loop interval
        dx1=[];dy1=[];avg_x=0;avg_y=0 
        for i in range(0,length): # Loop in single string
            if (i+interval) < length:
                dx1.append(int(X[i+interval] - X[i]) **2)
                dy1.append(int(Y[i+interval] - Y[i]) **2)
        #print(dx1,dy1)
        avg_x,avg_y = round(sum(dx1)/len(dx1),4),round(sum(dy1)/len(dy1),4)
        sol.append(avg_x + avg_y)
        #print(avg_x,avg_y)
    
    y = sol
    x = np.linspace(1,length-1,length-1)
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(x,y) 
    plt.title(f"{filename}")
    plt.xlabel("Time interval"); plt.ylabel("MSD")
    plt.axis([1,length,min(y)*0.8,max(y)*1.2])
    fig.savefig(f"{OutFolder}/Plot/{filename}.png")       
    if ImgShow ==True:
        plt.show()
    return sol

def MDD(X, Y,OutFolder,filename,ImgShow=False):
    sol=[];y=[]; length=len(X)
    for interval in range(1,length): # Loop interval
        dx1=[];dy1=[];avg_x=0;avg_y=0 
        for i in range(0,length): # Loop in single string
            if (i+interval) < length:
                dx1.append(int(X[i+interval] - X[i]) **2)
                dy1.append(int(Y[i+interval] - Y[i]) **2)
        #print(dx1,dy1)
        avg_x,avg_y = round(sum(dx1)/len(dx1),4),round(sum(dy1)/len(dy1),4)
        sol.append(avg_x + avg_y)
        #print(avg_x,avg_y)
   
    sns.set_theme(color_codes=True)
    g = plt.figure(figsize=(10,5))
    g =sns.regplot(x=np.linspace(1,length-1,length-1),
                y=sol, marker='o', label="example",
                robust=False, ci=95,
                scatter_kws={'s': 10, 'color':'#46b4b4'},
                line_kws={'lw': 2, 'color': '#b4466e'}) 
    g.figure.autofmt_xdate()
    plt.title(f'Linear Regression of {filename}')
    plt.xlabel("Time interval"); plt.ylabel("MSD")
    if ImgShow ==True:
        plt.show()
    plt.savefig(f"{OutFolder}/Plot/{filename}.png")  

    return sol