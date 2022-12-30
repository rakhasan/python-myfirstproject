# import cv2
# import pytesseract
#
# # image = cv2.imread('Whats.png')
# image = cv2.imread('numberplate1.png')
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray_three = cv2.merge([gray,gray,gray])
# thresh = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# # Blur and perform text extraction(you can use raw image)
# thresh = cv2.GaussianBlur(thresh, (3,3), 0)
# pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# data = pytesseract.image_to_string(thresh, lang='eng+ben', config='--psm 6')
# print(data)

# from PIL import Image
# from pytesseract import pytesseract
#
# # Defining paths to tesseract.exe
# # and the image we would be using
# path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# image_path = r"WhatsAppImage3.jpeg"
#
# # Opening the image & storing it in an image object
# img = Image.open(image_path)
#
# # Providing the tesseract executable
# # location to pytesseract library
# pytesseract.tesseract_cmd = path_to_tesseract
#
# # Passing the image object to image_to_string() function
# # This function will extract the text from the image
# text = pytesseract.image_to_string(img)
#
# # Displaying the extracted text
# print(text[:-1])

##################################################################################
# Import required packages
# import cv2
# import pytesseract
#
# # Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # Read image from which text needs to be extracted
# img = cv2.imread("WhatsAppImage1.jpeg")
#
# # Preprocessing the image starts
#
# # Convert the image to gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Performing OTSU threshold
# ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#
# # Specify structure shape and kernel size.
# # Kernel size increases or decreases the area
# # of the rectangle to be detected.
# # A smaller value like (10, 10) will detect
# # each word instead of a sentence.
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 70))
#
# # Applying dilation on the threshold image
# dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
#
# # Finding contours
# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_NONE)
#
# # Creating a copy of image
# im2 = img.copy()
#
# # A text file is created and flushed
# file = open("recognized.txt", "w+")
# file.write("")
# file.close()
#
# # Looping through the identified contours
# # Then rectangular part is cropped and passed on
# # to pytesseract for extracting text from it
# # Extracted text is then written into the text file
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#
#     # Drawing a rectangle on copied image
#     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Cropping the text block for giving input to OCR
#     cropped = im2[y:y + h, x:x + w]
#
#     # Open the file in append mode
#     file = open("recognized.txt", "a")
#
#     # Apply OCR on the cropped image
#     text = pytesseract.image_to_string(cropped)
#
#     # Appending the text into file
#     file.write(text)
#     file.write("\n")
#
#     # Close the file
#     file.close
#####################################################################
# specify path to the license plate images folder as shown below
# import pytesseract # this is tesseract module
# import matplotlib.pyplot as plt
# import cv2 # this is opencv module
# import glob
# import os
# # path_for_license_plates = os.getcwd() + "/license-plates/**/*.jpg"
# path_for_license_plates = os.getcwd() + "/license-plates/*.jpeg"
# # path_for_license_plates = "licenseplate.jpg"
# list_license_plates = []
# predicted_license_plates = []
#
# for path_to_license_plate in glob.glob(path_for_license_plates, recursive=True):
#     license_plate_file = path_to_license_plate.split("/")[-1]
#     license_plate, _ = os.path.splitext(license_plate_file)
#     '''
#     Here we append the actual license plate to a list
#     '''
#     list_license_plates.append(license_plate)
#
#     '''
#     Read each license plate image file using openCV
#     '''
#     img = cv2.imread(path_to_license_plate)
#
#     '''
#     We then pass each license plate image file
#     to the Tesseract OCR engine using the Python library
#     wrapper for it. We get back predicted_result for
#     license plate. We append the predicted_result in a
#     list and compare it with the original the license plate
#     '''
#     predicted_result = pytesseract.image_to_string(img, lang='eng',
#                                                    config='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
#     # predicted_result = pytesseract.image_to_string(img, lang='ben',
#     #                                                config='--oem 3 --psm 6 -c tessedit_char_whitelist = ১২৩৪৫৬৭৮৯০')
#     filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
#     predicted_license_plates.append(filter_predicted_result)
#
# print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy")
# print("--------------------", "\t", "-----------------------", "\t", "--------")
#
#
# def calculate_predicted_accuracy(actual_list, predicted_list):
#     for actual_plate, predict_plate in zip(actual_list, predicted_list):
#         accuracy = "0 %"
#         num_matches = 0
#         if actual_plate == predict_plate:
#             accuracy = "100 %"
#         else:
#             if len(actual_plate) == len(predict_plate):
#                 for a, p in zip(actual_plate, predict_plate):
#                     if a == p:
#                         num_matches += 1
#                 accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
#                 accuracy += "%"
#         print("	 ", actual_plate, "\t\t\t", predict_plate, "\t\t ", accuracy)
#
#
# calculate_predicted_accuracy(list_license_plates, predicted_license_plates)

############################################################################
import matplotlib.pyplot as plt
import numpy as np
import cv2

# image = cv2.imread('/content/picture.png')
# template = cv2.imread('/content/penguin.png')
# image = cv2.imread('uparrowfinder3.png')
# template = cv2.imread('uparrow.png')
# heat_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#
# h, w, _ = template.shape
# y, x = np.unravel_index(np.argmax(heat_map), heat_map.shape)
# cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
#
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
############################################################################
# import matplotlib.pyplot as plt
# import numpy as np
#
# # X axis parameter:
# xaxis = np.array([2, 8])
#
# # Y axis parameter:
# yaxis = np.array([4, 9])
#
# plt.plot(xaxis, yaxis)
# plt.show()
###############################################################
# import cv2
#
# img = cv2.imread('uparrowfinder3.png')
#
# cv2.rectangle(img, (10, 10), (100, 100), (0, 255, 0))
# cv2.rectangle(img, (120, 120), (150, 150), (255, 0, 0), 5)
# cv2.rectangle(img, (200, 200), (300, 400), (0, 0, 255), -1)
#
# cv2.imshow('image', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2

method = cv2.TM_SQDIFF_NORMED

# Read the images from the file
small_image = cv2.imread('uparrow.png')
large_image = cv2.imread('uparrowfinder2.png')

result = cv2.matchTemplate(small_image, large_image, method)

# We want the minimum squared difference
mn,_,mnLoc,_ = cv2.minMaxLoc(result)

# Draw the rectangle:
# Extract the coordinates of our best match
MPx,MPy = mnLoc

# Step 2: Get the size of the template. This is the same size as the match.
trows,tcols = small_image.shape[:2]

# Step 3: Draw the rectangle on large_image
cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

# Display the original image with the rectangle around the match.
cv2.imshow('output',large_image)

# The image is only displayed if we call this
cv2.waitKey(0)