import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('-a','--Animation', type=bool, default=True, help='Convert sequence of images into .mp4')
parser.add_argument('-m','--ExportMP4', type=bool, default=False, help='Export .mp4')
parser.add_argument('-g','--ExportGIF', type=bool, default=False, help='Export .gif')
args = parser.parse_args()
_ANI,_MP4,_GIF = args.ani, args.mp4, args.gif

from utils import *
#----Step1:ROOT folder/Envs setup---------------------#
Root1 = "./Brownian_motion"
Root2 = "./Export";  Root3 = f'{Root2}/TrackFile'
groups = ['1', '2', '3', '4', '5','Free','Laser'] # Named after exp. group
NUM = len(groups) # 7 experiments in total

MakeSubFolders(Root3, groups) # Create subfolders "ROI", "GIF", "Plot"
EnvSetup(f'{Root2}/RawData') # Create folder "RawData"

#----Step2:Convert to animation----------------------#
for index in groups[:NUM]:
    SrcFolder=f"{Root1}/{index}"
    OutFolder=f"{Root2}/RawData"
    if _ANI == True:
        IMG2MP4(SrcFolder, OutFolder, OutName=f'{index}', FPS=5)

#----Step3:Track through ROI-------------------------# 
for index in groups[:NUM]:
    SrcFolder1=f"{Root1}/{index}"
    SrcFolder2=f"{Root3}/{index}"
    OutFolder=f"{Root3}"
    
    X,Y = Track(SrcFolder1, OutFolder, OutName=f"{index}", SavePlot=True)
    # MSD(X ,Y, OutFolder, index, ImgShow=False)
    MDD(X ,Y, OutFolder, index, ImgShow=False)
    
    if _MP4 | _GIF == True:
        print(f"*Converting into animation...")
    if _MP4 == True:
        IMG2MP4(SrcFolder2,OutFolder,OutName=f'Track_{index}', FPS=5)
    if _GIF ==True:    
        PNG2GIF(SrcFolder2,OutFolder,OutName=f"Track_{index}",ImgFormat="png", duration=120)        

print("////"*18)
print(f"\nThis is the end of analysis! Check results in './Export'")