from flask import Flask, request, jsonify, json, make_response, redirect, session, send_from_directory
from flask_mysqldb import MySQL
import datetime
from flask_cors import CORS
from flask_mail import Mail,Message
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import asyncio
import secrets
import os, string, random
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import os
import numpy as np
import cv2
import imutils
UPLOAD_FOLDER = '/home/mks/Desktop/Metaorigin_labs/car_backend/uploads'
MODEL_FOLDER = '/home/mks/Desktop/Metaorigin_labs/car_backend/model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__, static_url_path='')
app.secret_key = secrets.token_urlsafe(16)
CORS(app)

export_file_url = 'https://drive.google.com/uc?export=download&id=1-6OmqUZtn7ESuldhZIXsu-_M5W7fEblb'
export_file_name = 'scratch_detector.pkl'

app.debug = True
app.env = 'development'
# app.config['SESSION_TYPE'] = 'filesystem'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'newuser'
app.config['MYSQL_PASSWORD'] = 'Root@123'
app.config['MYSQL_DB'] = 'meta_org'
app.config['MAIL_SERVER'] = 'smtp.gmail.com' #'server229.web-hosting.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'funto236@gmail.com'#'support@metaorigins.com'
app.config['MAIL_DEFAULT_SENDER'] = 'funto@gmail.com'#'support@metaorigins.com'
app.config['MAIL_PASSWORD'] = 'funto@123'#'UwuW[X0zSP^@'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mysql = MySQL(app)
mail = Mail(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/file/<path:path>')
def send_js(path):
    return send_from_directory('uploads', path)


@app.route('/login', methods=['POST'])
def login():
    try:
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            cur = mysql.connection.cursor()
            query = "SELECT password,first_name,last_name,id FROM users WHERE email ='"+email+"'"
            cur.execute(query)
            data = cur.fetchall()
            # print(data)
            if len(data) and check_password_hash(data[0][0], password):
                session['user_id'] = data[0][3]
                send_data = {'success': True, 'first_name': data[0][1], 'last_name': data[0][2], 'id': data[0][3]}
                print(session)
                return make_response(jsonify(send_data), 200)
            else:
                send_data = {'success': False}
                return make_response(jsonify(send_data), 200)
    except Exception as e:
        print(str(e))
        send_data = {'success': False}
        return make_response(jsonify(send_data), 200)


@app.route('/register', methods=['POST'])
def register():
    try:
        if request.method == 'POST':
            first_name = request.form['first_name']
            last_name = request.form['last_name']
            email = request.form['email']
            phone = request.form['phone']
            company = request.form['company']
            location = request.form['location']
            password = generate_password_hash('Meta@123')
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO users (first_name, last_name,password, email, phone, company, location,created_at) "
                        "VALUES (%s, %s ,%s, %s,%s, %s ,%s, %s)",
                        (first_name, last_name, password, email, phone, company, location, datetime.datetime.now()))
            mysql.connection.commit()
            cur.close()
            # msg = Message('Welcome', recipients=[email])
            # thread1 = threading.Thread(target=mail.send(msg))
            # thread1.start()
            message = "hi,"+first_name+"<br>you have successfully created account on Metaorigin Labs.your username "+email+", password is Meta@123.<br>thank you"

            subject = "welcome,"
            msg = Message(recipients=[email],
                          html=message,
                          subject=subject)
            # mail.send(msg)
            thread1 = threading.Thread(target=mail.send(msg))
            thread1.start()
            send_data = {'success': True}
            return make_response(jsonify(send_data), 200)
    except Exception as e:
        print(str(e))
        send_data = {'success': False}
        return make_response(jsonify(send_data), 200)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))

@app.route('/damage_check', methods=['POST'])
def damage_check():
    try:
        if request.method == 'POST':
            user_id = request.form['user_id']
            if not user_id:
                user_id = 1

            file = request.files['fileUpload']
            if file and allowed_file(file.filename):
                uid = randomString(10)
                filename = file.filename
                ext = filename.split(".")[-1]
                name = filename.split(".")[:-1]
                file_name = name[0]+'_'+uid+'.'+ext
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                orig_img = file_name

                learn = setup_learner()
                img = open_image(app.config['UPLOAD_FOLDER']+'/'+file_name)
                out_put = infer_imagewithcontours(img, learn)
                # print(out_put)
                ext = filename.split(".")[-1]
                name = filename.split(".")[:-1]
                file_name = name[0] + '_'+uid+'_' + '_final' + '.' + ext
                # img = open_image(app.config['UPLOAD_FOLDER']+'/'+file_name)
                # folder = os.path.join(app.config['UPLOAD_FOLDER'], img)
                out_put = cv2.cvtColor(out_put*255, cv2.COLOR_BGR2RGB)
                cv2.imwrite(app.config['UPLOAD_FOLDER']+'/'+file_name, out_put)
                process_img = file_name
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO images (user_id,original_img, process_img,created_at) "
                    "VALUES (%s, %s ,%s, %s)",
                    (user_id, orig_img, process_img, datetime.datetime.now()))
                mysql.connection.commit()
                cur.close()

            send_data = {'success': True, 'image': 'http://localhost:5000/file/'+file_name}
            return make_response(jsonify(send_data), 200)
    except Exception as e:
        print(str(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        send_data = {'success': False}
        return make_response(jsonify(send_data), 200)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(app.config['MODEL_FOLDER'], 'scratch_detector.pkl')
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

def acc_cars(input, target):
  target = target.squeeze(1)
  return (input.argmax(dim=1)==target).float().mean()
def infer_imagewithcontours(img,learn):
    mask,_,_ = learn.predict(img)
    # erode_kernel = np.ones((3,3))
    mask_copy = mask.data
    np_mask = np.asarray(mask_copy).reshape(448, 448)
    # np_mask = cv2.erode(np.uint8(np_mask), erode_kernel, 1)
    cnts = cv2.findContours(np.uint8(np_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    image_copy  = img.resize(448).data
    image_copy= np.asarray(image_copy)
    out= np.transpose(image_copy, (1, 2, 0))
    # for c in cnts:
    #   out = cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
    out = coordinate_bbox_generator(cnts, out)
    # print(type(out))
    # alpha = 0.3
    # out = alpha * np_mask.reshape(448,448,1) + (1-alpha) * out.get()
    # return out
    return out


def coordinate_bbox_generator(cnt, img):
    for box in cnt:
        x_coord, y_coord = [], []
        for item in box:
            x_coord.append(item[0][0])
            y_coord.append(item[0][1])
        r = x_coord
        c = y_coord
        # print(img)
        c1 = (int(min(r)), int(min(c)))
        # print(c1)
        c2 = (int(max(r)), int(max(c)))
        # print(c1)
        # print(c2)
        if c2[0]-c1[0] > 9 and c2[1] - c1[1] > 9:
            img = cv2.UMat(img).get()
            img = cv2.rectangle(img, c1, c2, (255, 0, 0), 1)
        # print(111111111)
        # print(img)
    return img

if __name__ == '__main__':
    app.run()
