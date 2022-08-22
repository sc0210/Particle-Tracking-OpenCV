from utils import Toolkits
import glob,os,os.path, numpy as np
import matplotlib.pyplot as plt

#----------------------ROOT folder setup---------------------#
Root1 = "./Brownian_motion"
Root2 = "./Export"
    
#----------------------Envs folder setup---------------------#
# Create folder "TrackFile"
root_dir = './Export/TrackFile'
subfolders = ('1', '2', '3', '4', '5','Free','Laser','ROI') # Named after group
Toolkits.makefolders(root_dir,subfolders)
    
# Create folde "RawData"
if os.path.exists("./Export/RawData") == False:
    os.mkdir("./Export/RawData")

#----------------------Convert to animation---------------------#
# Convert sequenes of images(.tif) into animantion(.mp4) -/RawData
Toolkits.TIF2AVI(InFolder=f"{Root1}/Laser", OutFolder=f"{Root2}/RawData", OutName='Laser', FPS=5)
Toolkits.TIF2AVI(InFolder=f"{Root1}/Free", OutFolder=f"{Root2}/RawData", OutName='Free', FPS=5)
for CaseNum in range(1,6):
    Toolkits.TIF2AVI(InFolder=f"{Root1}/Group1/{CaseNum}", OutFolder=f"{Root2}/RawData", OutName=f"Group1_{CaseNum}", FPS=5)


#-------------------------Track through ROI------------------------# 
Toolkits.Track(SrcFolder="./Brownian_motion/Laser", OutFoldName="Laser",SavePlot=True)
Toolkits.Track(SrcFolder="./Brownian_motion/Free", OutFoldName="Free",SavePlot=True)

for CaseNum in range(1,6):
    Toolkits.Track(SrcFolder=f"./Brownian_motion/Group1/{CaseNum}", OutFoldName=f"{CaseNum}",SavePlot=True)

#----------------------Convert to animationn---------------------#
# Convert sequenes of images(.jpg) into animantion(.mp4) -/TrackFile
print(f"\nConverting into animation...")
Toolkits.TIF2AVI(InFolder=f"{Root2}/TrackFile/Laser", OutFolder=f"{Root2}/TrackFile", OutName='Track_Laser', FPS=5)
Toolkits.TIF2AVI(InFolder=f"{Root2}/TrackFile/Free", OutFolder=f"{Root2}/TrackFile", OutName='Track_Free', FPS=5)
for CaseNum in range(1,6):
    Toolkits.TIF2AVI(InFolder=f"{Root2}/TrackFile/{CaseNum}", OutFolder=f"{Root2}/TrackFile", OutName=f"Group1_{CaseNum}", FPS=5)

print(f"\nFinish analysis! Check results in './Export'")