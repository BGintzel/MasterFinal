A Conda-Envirenment yml file is located in this folder for simplicity.
A list of needed packages can be found below



TRAINING EN FROM SCRATCH:

Run "main.py train 01" for demo training. The parameter "train" is used to start the training. The parameter 01 is used for the name of the network in case of many training runs.

It starts creating a dataset and training the Euler Network from scratch.

After that the training of the untrained Euler net begins. It trains for 5 Epochs and should achieve an accuracy of >98%


To save time you can download datasets and labels and put them in the "datasets" folder:
https://drive.google.com/drive/folders/1TXU-Yv7xTtPhwcaBxFJT_zm2YrUmpP5U?usp=sharing

Download an already trained Euler net from:
https://drive.google.com/drive/folders/1yY3Rbhc3sXVtI51hCvJCA8GJfvBWccBH?usp=sharing


USING OF EN:

Run "main.py use models/best01__checkpoint.pth 1.jpg 2.jpg" to run the EN on two images.
There are 4 Parameters:
- "use" is indicating the using the EN.
- relative path to Euler Net
- relative path to input image 1
- relative path to input image 2


TESTING ACCURACY OF EN:
Run "main.py check_accuracy models/best01__checkpoint.pth" to test the Accuracy of the EN on a newly created Test-Dataset.
- "check_accuracy" is indicating the checking the accuracy of the EN.
- relative path to Euler Net

Creating a new test data set takes some time (will upload one data set for use soon)


RUN ITERATIVE SELF-TRAINING ON AN EN:

Run "main.py IST models/best01__checkpoint.pth 5" to run iterative self-training for 5 iterations.
- "IST" is indicating running the iterative self-training on the EN.
- relative path to Euler Net
- number of iterations for self-training


FINDING AND PLOTTING UNINTENDED INPUTS (activation maximization optimized for label [0,1,0,0]:

Run "main.py find models/best01__checkpoint.pth 1" to find and plot unintended input-output pairs.
- "find" is indicating finding and ploting of unintended input-output pairs.
- relative path to Euler Net
- what kind of input
    - 1 incomplete
    - 2 arcs for full green and 180° red/blue
    - 3 colour
The results of these experiments are in the folder results

GENERAL REMARKS

All models (all training iterations) will be saved in folder "models".
Training stats are saved in folder "stats".

Euler net model architecture can be found in file "models.py".




List of dependencies:

Package           Version
----------------- -------
cycler            0.11.0
fonttools         4.33.3
kiwisolver        1.4.2
matplotlib        3.5.2
numpy             1.22.4
packaging         21.3
Pillow            9.1.1
pip               21.1.2
pyparsing         3.0.9
python-dateutil   2.8.2
setuptools        57.0.0
six               1.16.0
torch             1.11.0
tqdm              4.64.0
typing-extensions 4.2.0
wheel             0.36.2
