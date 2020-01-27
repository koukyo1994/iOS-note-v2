# Semester project - Handwritten Editor

## Prerequisite

* mac OSX 10.15.2 or higher
* Xcode 11.3.1 or higher
* (iPad with iPad OS 13.3 or higher)
* (Apple Pencil)

To reproduce the training you'll be also required to have

* Docker

## To reproduce training

1. Open `prototype/py/colab/Colab-create-dataset-and-train.ipynb` with Google Colaboratory.
2. Execute all the cell from the top. Note that it'll take about half a day to complete the training.
3. Download the weight `/content/iOS-note-v2/prototype/py/bin/3blocks_model_weights.h5` to your local repository of this and put that in `prototype/py/bin/`
4. Build a docker image with the command `make docker-build`. This command must be called in `prototype/py/`
5. After building the image, call `make env` and start up a docker container.
6. Use the command `python conversion.py --config config/resnet.3blocks.json` to get CoreML model weights.

## To reproduce the result from Application

1. Open `prototype/NoteApp/NoteApp.xcodeproj` with Xcode.
2. Compile and run on real device.

In case you do not have iPad and Apple pencil, this application offers a way to do that with simulator. After starting up the App, click the button in the left of Navigation Bar that says `Finger / Stylus`. When it is set to `Finger`, you can use the App on simulator.
