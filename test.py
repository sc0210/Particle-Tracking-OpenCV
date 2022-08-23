import glob
from PIL import Image
def MakeGIF(ImgFormat, SrcFolder, OutName, duration=120):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{SrcFolder}/*.{ImgFormat}"))]
    frame_one = frames[0]
    frame_one.save(f"{OutName}.gif", format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)
    
if __name__ == "__main__":
    ImgFormat = "png"
    SrcFolder="./Export/TrackFile/Laser"
    OutName="Laser"
    MakeGIF(ImgFormat, SrcFolder, OutName, duration=120)