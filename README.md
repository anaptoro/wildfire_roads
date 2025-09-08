This repository aims to be an end to end workflow for initial analysis on tree crops overlapping with highways. Some applications that could benefit from this wildfire prevention, risk reduction tree pruning, prevention of animal run over. The inputs here were NAIP images and OSM highways, to detect tree crowns the DeepForest model was used.

Usage:
Its recommended that you run the application using the existing container. So, first you should run:

docker build -t wildfire-pl:cpu 
docker compose up -d wildfire 

Inside the container the running command is:

python run.py --aoi Lake_Oswego_OR --out outputs
(Lake_Oswego is an example AOI, run that as a test, you can check other example AOIs in the config file)

The suggested command for real applications is: 
python run.py --bbox -122.709 45.408 -122.676 45.431 --out outputs

All the parameters used in the DeepForest model and the final risk map can be tunned in the config.py file

Please notice that probably calibration is needed depending on geography.