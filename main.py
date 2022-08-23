from utils import *
import glob,os,os.path, numpy as np
import matplotlib.pyplot as plt

#----------------------ROOT folder setup---------------------#
Root1 = "./Brownian_motion"
Root2 = "./Export"
    
#----------------------Envs folder setup---------------------#
# Create folder "TrackFile"
root_dir = './Export/TrackFile'
groups = ['1', '2', '3', '4', '5','Free','Laser'] # Named after group
makefolders(root_dir,subfolders=groups)
# Create folder "RawData"
if os.path.exists("./Export/RawData") == False:
    os.mkdir("./Export/RawData")

#----------------------Convert to animation---------------------#
# Convert sequenes of images(.tif) into animantion(.mp4) -/RawData
for index in groups:
    SrcFolder=f"{Root1}/{index}"
    OutFolder=f"{Root2}/RawData"
    IMG2MP4(SrcFolder, OutFolder, OutName=f'{index}', FPS=5)

#-------------------------Track through ROI------------------------# 
for index in groups:
    SrcFolder=f"./Brownian_motion/{index}"
    OutFolder=f"{Root2}/TrackFile"
    Track(SrcFolder, OutFolder, OutName=f"{index}", SavePlot=True)

#----------------------Convert to animationn---------------------#
# Convert sequenes of images(.jpg) into animantion(.mp4/.gif) -/TrackFile
print(f"\nConverting into animation...")
for index in groups:
    SrcFolder=f"{Root2}/TrackFile/{index}"
    OutFolder=f"{Root2}/TrackFile"
    IMG2MP4(SrcFolder, OutFolder, OutName=f'Track_{index}', FPS=5)
    PNG2GIF(SrcFolder, OutFolder, OutName=f"Track_{index}",ImgFormat="png", duration=120)

print("////"*18)
print(f"\nThis is the end of analysis! Check results in './Export'")