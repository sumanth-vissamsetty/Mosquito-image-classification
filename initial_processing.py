import os, shutil, os.path
from pathlib import Path
import image_slicer

#Setting the path.
raw_images_path = Path("C:\\Users\\svissa1\\Documents\\research\\data\\raw-data\\cropped_by_factor_20\\validation\\pupae")
raw_images_path_string = str(raw_images_path)

#Checking if the folder contains images or not.
for file in os.listdir(raw_images_path_string):
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        isContainingImage = True
        #print("Found an image:\t" + file)
        break

#Creating a new folder "processed_images", same-level as the "raw_images".
if isContainingImage == True:
    processed_images_path = raw_images_path.parent
    processed_images_path = os.path.join(str(processed_images_path),"processed_images")
    if os.path.exists(processed_images_path):
        x = input('Do you want to over-write the previous execution?\nPress \'Y\' if you would like to do so.\n')
        if x.strip() == 'Y' or x.strip() == 'y':
            print("\nCreating the new folder...")
            #Deleting the existing folder.
            shutil.rmtree(processed_images_path, ignore_errors=True)
            #and creating a new empty one.
            os.makedirs(processed_images_path)
            if os.path.isdir(processed_images_path):
                processed_images_path = os.path.join(processed_images_path, "")
            print("Completed creating.")
    else:
        print("\nCreating the folder")
        os.makedirs(processed_images_path)
        if os.path.isdir(processed_images_path):
                processed_images_path = os.path.join(processed_images_path, "")

#Cropping the images and saving them into the new folder.
print("\nStarting to process the images")
print("Processing the images...")
for file in os.listdir(raw_images_path_string):
    if file.endswith(".png") or file.endswith(".jpg"): #or file.endswith(".jpeg"):
        #path where the image is present
        image_path = os.path.join(raw_images_path_string, file)
        #tiles = image_slicer.slice(image_path, 20, save = False)
        tiles = image_slicer.slice(image_path, 25, save = False)
        #path where the image should be saved after processing
        sliced_image_path = os.path.join(processed_images_path, file)
        if os.path.exists(sliced_image_path):
            shutil.rmtree(sliced_image_path, ignore_errors=True)
            #and creating a new empty one.
            os.makedirs(sliced_image_path)
            if os.path.isdir(sliced_image_path):
                sliced_image_path = os.path.join(sliced_image_path, "")
        else:
            os.makedirs(sliced_image_path)
            if os.path.isdir(sliced_image_path):
                sliced_image_path = os.path.join(sliced_image_path, "")
        image_slicer.save_tiles(tiles, directory=sliced_image_path, prefix= file)
        
    
print("\nFinished processing. You can open the 'processed_images' folder for results.")