This script detects width and water level of a bottle via webcam.
Detects bottle using YOLO and detects the water level using edge detection given than the level is between 80% of the bottle.

#IMPORTANT
To install the required libraries, open Command Prompt/Terminal and navigate to this folder. Then run "pip install requirements.txt".
Do NOT delete the 'yolo' folder under any circumstances. The weights are custom trained and have no alternatives.


#PROCEDURE
Make sure that python is added to the PATH and all the requirements are installed.
To run the script first open a Command Prompt/Terminal window and navigate to the project folder.
Then type "python detect.py" to start the script.
Select between webcam and static image when asked.
If webcam selected align the bottle between the blue lines.
Make sure that the water level is as level with the camera as possible.
Press 'd' once aligned.
If static image selected enter the full path of the image. (e.g. C:/Users/Username/Desktop/image.jpg)
Enter the height of the bottle.
If no bottle was detected, or the water level was not detected, try again by varying the light and the angle.
If detected press 'q' to close the image.
Press Ctrl+C to stop the script.
