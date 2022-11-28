import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('-a','--Animation',type=int,default=True, help='Convert sequence of images into .mp4')
parser.add_argument('-s','--SaveFig',type=int,default=True, help='Save plot results')
parser.add_argument('-v','--ExportVid',type=int,default=True, help='Export .mp4 & .gif')
args = parser.parse_args()
_ANI,_Plot,_Export = args.Animation, args.SaveFig, args.ExportVid

from utils import *

#----Step1:ROOT folder/Envs setup---------------------#
Root1 = "./Brownian_motion"
Root2 = "./Export"
Root3 = "./Export/TrackFile/Reserved"
folders = ['RawData','TrackFile']
subfolders = ["ROI", "GIF", "Plot" ,"Reserved"]
groups = ['1', '2', '3', '4', '5','Free','Laser'] # Named after exp. group
NUM = len(groups) # 7 experiments in total

EnvSetup(f'./Export')
MakeSubFolders(f'./Export', folders) # Create subfolders groups "RawData","TrackFile"
MakeSubFolders(f'./Brownian_motion', groups) # Create subfolders groups "1", "2","3"...
MakeSubFolders(f'./Export/TrackFile', subfolders) # Create subfolders "ROI", "GIF", "Plot" ,"Reserved"

#----Step2:Convert to animation----------------------#
for index in groups[1:2]:
    SrcFolder = f"{Root1}/{index}"
    OutFolder =f"{Root2}/RawData"
    # if _ANI == True|1:
        # IMG2MP4(SrcFolder, OutFolder, OutName=f'{index}', FPS=5)

#----Step3:Track through ROI-------------------------# 
METHOD = 1; FPS=10
_SrcFolder = "./Brownian_motion" # import analysis file
_DstFolder = "./Export/TrackFile/Reserved" # Saved tracking result by frame

for index in groups[:]:
    if METHOD == 1:
        X,Y = Track(_SrcFolder, _DstFolder, GroupIndex=index)   
    elif METHOD == 2:
        X,Y = Track2(_SrcFolder, _DstFolder, GroupIndex=index)
    else:
        print("Please check METHOD")
    
    MSD(X ,Y, FPS, "./Export/TrackFile", GroupIndex=index, ImgShow=True)
    
    if _Export  == True|1:
        #print(f"=> Converting into animation...")
        Src = f"{Root3}/{index}"
        FilenName = f"Track_{index}"
        #IMG2MP4(Src, f"{Root2}/{index}", FilenName, FPS=20)
        #PNG2GIF(Src, f"{Root2}/TrackFile/", FilenName, ImgFormat="png", duration=10)        

print("////"*18)
print(f"\nThis is the end of analysis! Check results in './Export'")