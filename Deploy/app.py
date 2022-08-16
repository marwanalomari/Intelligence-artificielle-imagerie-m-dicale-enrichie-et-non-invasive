import os
import uuid
import flask
import urllib
from PIL import Image
from flask import Flask, render_template, request, send_file
import requests
import matplotlib.pyplot as plt
from model import define_G
import torch
import numpy as np
import cv2

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = './weight/FULL/generator_t2_tumor_bw.pth'
gen = define_G(4, 1, 64, 'unet_128', norm='instance', )
gen.load_state_dict(torch.load(model_path))
gen.cuda()
gen.eval()


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename):
    img = Image.open(filename).convert('L').rotate(90)
    t2 = []
    mm0 = np. array(img)
    mm0= cv2.resize(mm0,(128,128),interpolation=cv2.INTER_AREA)
    mm0=mm0/mm0.max()
    t2.append(mm0)
    t2 = np.array(t2)
    t2 = t2[0, :, :]
    t2 = ((torch.from_numpy(t2)[np.newaxis,np.newaxis,:]-0.5)/0.5).float().cuda()
    ## predict flair using t2
    c=torch.zeros(1,3).cuda()
    c[np.arange(t2.size(0)),0]=1
    f_pred = (gen(t2,c).squeeze().data.cpu().numpy()+1)/2

    ## predict t1ce using t2
    c=torch.zeros(1,3).cuda()
    c[np.arange(t2.size(0)),1]=1
    t1ce_pred = (gen(t2,c).squeeze().data.cpu().numpy()+1)/2

    ## predict t1 using t2
    c=torch.zeros(1,3).cuda()
    c[np.arange(t2.size(0)),2]=1
    t1_pred = (gen(t2,c).squeeze().data.cpu().numpy()+1)/2
    
    t2_S = []
    mm0_S = np. array(img)
    mm0_S= cv2.resize(mm0,(128,128),interpolation=cv2.INTER_AREA)
    mm0_S=mm0/mm0.max()
    t2_S.append(mm0_S)
    t2_S = np.array(t2_S)
    t2_S = t2_S[0, :, :]
    
    return t2_S , f_pred, t1ce_pred, t1_pred




@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename
    
                t2_S , f_pred, t1ce_pred, t1_pred = predict(img_path)
                plt.figure(figsize=[11, 10])
                plt.imshow(t2_S,'gray')
                plt.savefig('static/images/t2_S.jpg',format='jpg')
                
                plt.imshow(f_pred,'gray')
                plt.savefig('static/images/f_pred.jpg',format='jpg')
                
                plt.imshow(t1ce_pred,'gray')
                plt.savefig('static/images/t1ce_pred.jpg',format='jpg')
                
                plt.imshow(t1_pred,'gray')
                plt.savefig('static/images/t1_pred.jpg',format='jpg')                
            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                t2_S , f_pred, t1ce_pred, t1_pred = predict(img_path)
                plt.figure(figsize=[11, 10])

                plt.imshow(t2_S,'gray')
                plt.savefig('static/images/t2_S.jpg',format='jpg')
                
                plt.imshow(f_pred,'gray')
                plt.savefig('static/images/f_pred.jpg',format='jpg')
                
                plt.imshow(t1ce_pred,'gray')
                plt.savefig('static/images/t1ce_pred.jpg',format='jpg')
                
                plt.imshow(t1_pred,'gray')
                plt.savefig('static/images/t1_pred.jpg',format='jpg')  

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)

