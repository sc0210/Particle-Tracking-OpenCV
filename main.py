from utils import Toolkits
import glob, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

if "__name__" == "__main__":
    ## Folder Path
    Root1 = "./Brownian_motion"
    Root2 = "./Export"
    
    #----------------------Convert to animationn---------------------#
    # Convert sequenes of images(.tif) into animantion(.mp4) -/RawData
    Toolkits.TIF2AVI(InFolder=f"{Root1}/Laser", OutFolder=Root2, OutName='Laser', FPS=5)
    Toolkits.TIF2AVI(InFolder=f"{Root1}/Free", OutFolder=Root2, OutName='Free', FPS=5)
    for CaseNum in range(1,6):
        Toolkits.TIF2AVI(InFolder=f"{Root1}/Group1/{CaseNum}", OutFolder=Root2, OutName=f"Group1_{CaseNum}", FPS=5)


    #-------------------------Track through ROI------------------------# 
    Toolkits.Track(SrcFolder="./Brownian_motion/Laser", OutFoldName="Laser",SavePlot=True)
    Toolkits.Track(SrcFolder="./Brownian_motion/Free", OutFoldName="Free",SavePlot=True)

    for CaseNum in trange(1,6):
        Toolkits.Track(SrcFolder=f"./Brownian_motion/Group1/{CaseNum}", OutFoldName=CaseNum,SavePlot=True)

    #----------------------Convert to animationn---------------------#
    # Convert sequenes of images(.jpg) into animantion(.mp4) -/TrackFile
    Toolkits.TIF2AVI(InFolder=f"{Root2}/TrackFile/Laser", OutFolder=f"{Root2}/TrackFile", OutName='Track_Laser', FPS=5)
    Toolkits.TIF2AVI(InFolder=f"{Root2}/TrackFile/Free", OutFolder=f"{Root2}/TrackFile", OutName='Track_Free', FPS=5)
    for CaseNum in range(1,6):
        Toolkits.TIF2AVI(InFolder=f"{Root2}/TrackFile/{CaseNum}", OutFolder=f"{Root2}/TrackFile", OutName=f"Group1_{CaseNum}", FPS=5)
