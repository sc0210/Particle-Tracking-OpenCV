# Particle-Tracking
> This project is inspired by a lecture (Brownian-motion-exp.) in school. The main goal of this project is to extend self-learning programing skill and provide more convenient tools to optimized the postprocessing. Looking forward to share with junior students! 

## How to apply this tool at local machine (Steps by steps)
1. Clone this repository to your folder `git clone https://github.com/samchen0210/Particle-Tracking.git`
2. Install package used this project `pip install -r requirements`
3. Organize input data (tif images) by group and store in respective folder 
4. Execute main program `python main.py`
5. Check up the result in `./Export`

## Check (to-do) list  <sub>***(last updated 8/22***)</sub>
- [ ] Part 1 Develop tools with funcitons listed bellow 
  - [x] Read several types(tif, jpg, png) of image `ReadGrayImg(RscPath, show=False)`
  - [x] Convert sequences of images into animation `TIF2AVI(InFolder,OutFolder="./Export",OutName="test",FPS=5)`
  - [ ] Image preprocessing (kernel/ filter) (edge detection/ blur/ sharpen/ fill)
  - [x] Relation beetween sequentail of images 
  - `normxcorr2(template, image, mode="full")`
  - `Track(SrcFolder, OutFoldName="test" ,SavePlot=True)`
  - [ ] Coefficient of viscosity
  - [ ] Graph the in XY cororidnated system
- [ ] Part 2 Organized and record the process
  - [ ] Github -> *Create this repository!* 
  - [ ] TA (teaching material, demo code, ppt)

## Project Workflow
1. Collect data and necessary environment setting (temp.)
2. Preprocesisng
   - Convert to animation -> find the trend of particles movement
   - Track -> Normal Cross Correlaiton
3. Play around with computer vision tools in "OpenCV"
   - Common filter
   - Color transform
   - Resize, Augmentation
