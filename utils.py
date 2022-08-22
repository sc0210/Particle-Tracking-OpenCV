import cv2, os, glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import fftconvolve
from functools import partial



class Toolkits:
    def makefolders(root_dir, subfolders):
        concat_path = partial(os.path.join, root_dir)
        for subfolder in map(concat_path, subfolders):
            os.makedirs(subfolder, exist_ok=True)  # Python 3.2+

    def ReadGrayImg(RscPath, show=False):
        """
        Read an image.
        
        Args:
            RscPath:(str)image path
            show:(bool)show the image on the screen
        """
        Img = cv2.imread(RscPath,0)
        if show ==True:
            Img.show()
        return Img

    def TIF2AVI(InFolder,OutFolder="./Export",OutName="test",FPS=5):
        """
        Convert sequential TIFs images into AVI clip format.
        
        Args:
            InFolder: (str) input folder locaiton (picture file)
            OutFolder: (str) output folder location (video file)
            OutName: (str) AVI clip name
            FPS: (int) frame per seonds
        """

        FPS = int(FPS) # Frame per seconds
        InFolder = str(InFolder); OutFolder = str(OutFolder)
        video_name = f'{OutFolder}/{OutName}.mp4'
            
        # Read every .Tif files in the folder

        ImgType = os.path.splitext(os.listdir(InFolder)[0])[1]
        if ImgType == ".tif":
            images = [img for img in os.listdir(InFolder) if img.endswith(".tif")]
        elif ImgType == ".png":
            images = [img for img in sorted(os.listdir(InFolder)) if img.endswith(".png")]
        else:
            print("Not recongized image type")
       
        # Fetch the frame shape(e.g height, width, layer) 
        frame = cv2.imread(os.path.join(InFolder, images[0]))
        height, width, layers = frame.shape
            
        # Adjust output location
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_name, fourcc, FPS, (width,height))
            
        # Convert images(.tif) into video(.avi)
        for image in images:
            video.write(cv2.imread(os.path.join(InFolder, image)))
            
        cv2.destroyAllWindows()
        video.release()

    def normxcorr2(template, image, mode="full"):
        from scipy.signal import fftconvolve
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
        aux = Toolkits.dog(img,sigma=sigma,k=k,gamma=gamma)/255
        for i in range(0,aux.shape[0]):
            for j in range(0,aux.shape[1]):
                if(aux[i,j] < epsilon):
                    aux[i,j] = 1*255
                else:
                    aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
        return aux

    def xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
        aux = Toolkits.dog(img,sigma=sigma,k=k,gamma=gamma)/255
        for i in range(0,aux.shape[0]):
            for j in range(0,aux.shape[1]):
                if(aux[i,j] >= epsilon):
                    aux[i,j] = 1
                else:
                    ht = np.tanh(phi*(aux[i][j] - epsilon))
                    aux[i][j] = 1 + ht
        return aux*255

    def Track(SrcFolder, OutFoldName="test" ,SavePlot=True):
        Corr =[0,0,0,0,0,0,0]
        list_x ,list_y = [],[]

        #---------ROI section--------------------------------------------------------------------------------
        FileNumber = len(os.listdir(SrcFolder))
        ROI_INDEX = np.random.randint(0,FileNumber)

        template = Toolkits.ReadGrayImg(f"{SrcFolder}/{sorted(os.listdir(SrcFolder))[ROI_INDEX]}", False)        
        output = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        blur = cv2.medianBlur(template,5)  
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,1,20,param1=50,param2=10,minRadius=3,maxRadius=20)
        print("////"*18)
        print(f"Currenet analysis image: {OutFoldName}")
        if circles is not None:
            # Get the (x, y, r) as integers
            circles = np.uint16(np.around(circles))
            #print(circles)
            # loop over the circles
            for i in circles[0,:]:
                cv2.circle(output,(i[0],i[1]),i[2],(100, 103, 227),-1) # draw outer
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

                print(f"(CenterX,CenterY):({i[0]},{i[1]}), (x1,y1):({x1},{y1}), (x2,y2):({x2},{y2}), radius:{i[2]}")

        
        cv2.imwrite(f"./Export/TrackFile/ROI/{OutFoldName}_ROI.jpeg",output)
        roi = template.copy()
        cv2.circle(roi,(Corr[4],Corr[5]),Corr[6],(255, 0, 0),-1)
        roi = np.asarray(roi)[Corr[1]:Corr[3], Corr[0]:Corr[2]] #26:44,18:34]# opposite input seleciton
        roi = cv2.GaussianBlur(roi,(5,5),0)  
        #roi = Toolkits.xdog_garygrossi(roi,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10)

        #---------NormalXCorr--------------------------------------------------------------------------------
        for filename in sorted(os.listdir(SrcFolder)):
            #num1, num2 = str(index).zfill(4), str(index+1).zfill(4)
            image = Toolkits.ReadGrayImg(f"{SrcFolder}/{filename}", False)
            imarray1 = np.asarray(image)
            imarray1 = cv2.GaussianBlur(imarray1,(7,7),0)
            #kernel = np.ones((3,3), np.uint8)
            #imarray1 = cv2.erode(imarray1, kernel, iterations = 1)
            #imarray1 = cv2.dilate(imarray1, kernel, iterations = 1)
            #imarray1 = Toolkits.xdog_garygrossi(imarray1,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10)
                    
            #Cross correlation
            corr = Toolkits.normxcorr2(roi, imarray1, mode="same")
            y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
            #print(f"x:{x},y:{y}")
            list_x.append(x),list_y.append(y)
            
            if SavePlot == True:
                fig, (ax_orig, ax_corr) = plt.subplots(1, 2)

                ax_orig.imshow(imarray1, cmap='gray')
                ax_orig.set_title(f'{filename}(Image)')
                ax_orig.plot(x, y, 'ro',linewidth=2, markersize=12)
                ax_orig.set_axis_off()

                ax_corr.imshow(roi, cmap='gray')
                ax_corr.set_title(f'No.{ROI_INDEX}(ROI/Template)')
                ax_corr.set_axis_off()
                fig.tight_layout()
                filename = filename.replace(".tif","")
                plt.savefig(f"./Export/TrackFile/{OutFoldName}/{filename}.png", bbox_inches='tight')
                plt.cla()
                plt.close(fig)
                
        print(f"image shape:{imarray1.shape}, ROI:{roi.shape}\n")
                
        return list_x, list_y