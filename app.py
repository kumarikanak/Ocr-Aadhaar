from flask import Flask, jsonify, request
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
import os
import re
import cv2
import urllib.request
import pandas as pd
import requests
import json
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowPredictor
from sagemaker.predictor import JSONSerializer, JSONDeserializer
from skimage import io
from skimage import img_as_ubyte
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imshow, show
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.stats import mode



app = Flask(__name__)

@app.route('/ocr_aadhar',methods=['POST'])


def image_check():


    endpoint_name = 'tensorflow-training-2021-12-02-17-57-59-841'
    client = boto3.client('runtime.sagemaker')
    predictor = TensorFlowPredictor(endpoint_name)
    end_point = predictor.endpoint_name
    #print(end_point)

    data = json.loads(request.data)
    img_url = data.get("data",None)
    # if url is None:
    #     return jsonify({"message":"url not found"})
    # else:
    #     return (url)
    urllib.request.urlretrieve(img_url,"test.jpg")
    img = Image.open("test.jpg")
    size_img = img.size[:2]
    print(str(size_img))
    input_image = np.array(img.resize((128, 128))) / 256.
    input_image = np.expand_dims(input_image, axis=0)
    #print(input_image.shape)
    img = np.array(img)

    #Predict the masked image for the given image
    img_data = {"instances": input_image.tolist()}
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    out_img = predictor.predict(img_data)
    out_mask = np.argmax(out_img['predictions'][0], axis=-1)
    out_mask = out_mask[..., np.newaxis]
    mask_arr = out_mask
    #print(type(mask_arr))

    #Resize masked image to the original size of the given image
    resize_mask = cv2.resize(mask_arr,(img.shape[1],img.shape[0]),interpolation= cv2.INTER_LINEAR_EXACT)
    np.unique(resize_mask)
    #print(resize_mask.shape)

    img_new = np.copy(img)
    img_new[resize_mask==0] =0

    #get bounding box around the id card
    mask_rgb = np.array(Image.fromarray(np.squeeze(resize_mask*255).astype(np.uint8)).convert('RGB'))
    imgray = cv2.cvtColor(mask_rgb,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
        # find contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w * h
        areas.append(area)
        boxes.append([x, y, w, h])

    x, y, w, h = boxes[areas.index(sorted(areas)[-1])]
    output_img = img[y:y+h, x:x+w]

    img = Image.fromarray(output_img)

    # out_image = img.copy()

    # MAX_SIZE = (300, 300)
    # img.thumbnail(MAX_SIZE)
    # # creating thumbnail
    # print(img.size)
    out_img = BytesIO()
    img.save(out_img, format='jpeg')
    out_img.seek(0)

    #Save the cropped image in S3 Bucket
    bucket='kanak-sagemaker'
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).put_object(Key='cropped_aadhar_f.jpg',Body=out_img,ContentType='image/jpg',ACL='public-read')
    photo_c = 'cropped_aadhar_f.jpg'


                                    #Correct Orientation of Cropped Aadhar
    s3_object = s3.Object(bucket,photo_c)
    s3_response = s3_object.get()
    stream = BytesIO(s3_response['Body'].read())
    orig_pil = Image.open(stream)
    orig = np.array(orig_pil)
    #print(orig)
    image = rgb2gray(orig)
    edges = canny(image)

    # Classic straight-line Hough transform between 0.1 - 360 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 360.0))
    
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    print(skew_angle)
    
    #final_img = rotate(orig, skew_angle, cval=1,resize=True)
    final_img = img_as_ubyte(rotate(orig, skew_angle, cval=1,resize=True))
    rotate_img = Image.fromarray(final_img)
    out_img = BytesIO()
    rotate_img.save(out_img, format='jpeg')
    out_img.seek(0)
    s3.Bucket(bucket).put_object(Key='rotated_aadhar_f.jpg',Body=out_img,ContentType='image/jpg',ACL='public-read')
    photo_r = 'rotated_aadhar_f.jpg'


                                                            #Detect Text from Aadhar
    client=boto3.client('rekognition')

    response=client.detect_text(Image={'S3Object':{'Bucket':bucket,'Name':photo_r}})
                        
    textDetections=response['TextDetections']
    print ('Detected text\n----------')
    df = pd.DataFrame(response['TextDetections'])
    result_df = pd.DataFrame(columns=['Text','Id','Type','Confidence'])
    df.drop(['Geometry','ParentId'], axis = 'columns',inplace = True)
    #df = df.loc[(df['Type'] != 'WORD') & (df['Confidence'] > 85)]
    df = df.loc[(df['Type'] != 'WORD') & (df.DetectedText.str.contains("Address|ADDRESS|DOB|Date Of Birth|Year of Birth|Male|Female|MALE|FEMALE") | (df['Confidence'] > 85))]
    #print(df)
    df = df[~df.DetectedText.str.contains('AADHAAR|aadhaar|VID|HRA ARФR|- well|1800 300 1947|1947|help|www|www.|Bengaluru-560 001|P.O.')]
    print(df)

    temp = []
   
    for row in df.itertuples():
        #print(row)
        text = getattr(row, 'DetectedText')
        #print(text)
        
        #to remove noise and unnecessary characters from string
        if text != ' ' or text != '  ' or text != '':
            text = re.sub('[^A-Za-z0-9-/,.() ]+', '', text)
            text = text.strip()
            text = re.sub(r'\s{2,}', ' ', text) 
        #print(text)
        temp.append(text)
    details = pd.unique(temp).tolist()
    
    imp = {}
  
    #Name
    for idx in range(len(details)):
        # if 'GOVERNMENT' in details[idx].upper() or 'OF' in details[idx].upper() or 'INDIA' in details[idx].upper():
        if 'GOVERNMENT OF INDIA' in details[idx].upper() :
            imp["Name"] = details[idx + 1]
        


    #Aadhar Number
    for idx in range(len(details)):
        if re.search(r"[0-9]{4}\s[0-9]{4}\s[0-9]{4}", details[idx]):
                try:
                    imp['Aadhar No'] = re.findall(r"[0-9]{4}\s[0-9]{4}\s[0-9]{4}", details[idx])[0]
                except Exception as _:
                    imp['Aadhar No'] = "Not Found"
        
    #Address
    for idx in range(len(details)):
         #if 'GOVERNMENT' in details[idx].upper() or 'OF' in details[idx].upper() or 'INDIA' in details[idx].upper():
        if 'UNIQUE IDENTIFICATION AUTHORITY OF INDIA' in details[idx].upper() :
            address = details[idx + 1:]
            try:
                if 'Address' in address[0] and address[0].split('Address', 1)[1].strip() == '':
                    imp["Address"] = address[1]
                    for line in address[2:]:
                        imp["Address"] += ' ' + line
                    imp['Address'] = imp['Address'].strip()
                    
                elif 'Address' in address[0] and address[0].split('Address', 1)[1].strip() != '':
                    imp["Address"] = address[0].split('Address', 1)[1].strip()
                    for line in address[1:]:
                        imp["Address"] += ' ' + line
                    imp['Address'] = imp['Address'].strip()
                    
#                 if 'Address' in address[0]:
#                     if address[0].split('Address', 1)[1].strip() != '':
#                         imp["Address"] = address[0].split('Address', 1)[1].strip()
#                     elif address[0].split('Address', 1)[1].strip() == '':
#                         imp["Address"] = address[1]
#                     for line in address[1:]:
#                         imp["Address"] += ' ' + line
#                     imp['Address'] = imp['Address'].strip()
                    
                elif 'Address' in address[1]:
                    if address[1].split('Address', 1)[1].strip() != '':
                        imp["Address"] = address[1].split('Address', 1)[1].strip()
                    for line in address[2:]:
                        imp["Address"] += ' ' + line
                    imp['Address'] = imp['Address'].strip()
        
                elif 'Address' in address[2]:
                    if address[2].split('Address', 1)[1].strip() != '':
                        imp["Address"] = details[2].split('Address', 1)[1].strip()
                    for line in address[3:]:
                        imp["Address"] += ' ' + line
                    imp['Address'] = imp['Address'].strip()
        
                else:
                    imp["Address"] = 'Failed to read Address'
            except Exception as _:
                imp["Address"] = 'Failed to read Address'
    
    #PinCode
    for idx in range(len(temp)):
        if re.search(r"[1-9]{1}[0-9]{5}", temp[idx]):
            try:
                imp['PinCode'] = re.findall(r"[1-9]{1}[0-9]{5}", temp[idx])[0]
                #details.remove(temp[idx])
            except Exception as _:
                imp['PinCode'] = "Not Found"
                
    #DOB or YOB
    for idx in range(len(temp)):
        if "DOB" in temp[idx]:
                # if string similar to date is found, use it as a hook to find other details
                if re.search(r"[0-9]{2}\-|/[0-9]{2}\-|/[0-9]{4}", temp[idx]):
                    try:
                        imp["Date of Birth"] = re.findall(r"[0-9]{2}\-[0-9]{2}\-[0-9]{4}", temp[idx])[0]
                    except Exception as _:
                        imp["Date of Birth"] = re.findall(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", temp[idx])[0]
                    
                    
    for idx in range(len(temp)):
        # handle variation of 'Year of Birth' in place of DOB 
        if "Year of Birth" in temp[idx]:
                # handle variation of 'Year of Birth' in place of DOB
                try:
                    imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
                except Exception as _:
                    imp["Year of Birth"] = "Not Found"
        
                
    #Gender
    for idx in range(len(temp)):
        if temp[idx].endswith("Female"):
            imp["Gender"] = "Female"
        elif temp[idx].endswith("Male"):
            imp["Gender"] = "Male"
        elif temp[idx].endswith("FEMALE"):
            imp["Gender"] = "Female"
        elif temp[idx].endswith("MALE"):
            imp["Gender"] = "Male"


    os.remove("test.jpg")

    return jsonify(imp)

if __name__ == "__main__":
    app.run(debug=True)