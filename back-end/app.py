import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *
from processor.AIDetector_pytorch import Detector,FusionDetectot
from functools import wraps
import core.main
from flask_cors import CORS

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app = Flask(__name__)

# 宿主机能telnet通,但http访问不了后端,可能是跨域问题,添加下列代码没用,
# 将app.run中地址从本地localhost改为0.0.0.0通配地址,应用程序能够接收来自任何网络接口的连接请求
CORS(app, resources=r'/*')

app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    # response.headers['Access-Control-Allow-Headers'] = 'Referer,Accept,Origin,User-Agent,Access-Control-Allow-Origin,Content-Type,X-Requested-With'
    return response

# 解决跨域
# def allow_cross_domain(fun):
#     @wraps(fun)
#     def wrapper_fun(*args, **kwargs):
#         rst = make_response(fun(*args, **kwargs))
#         rst.headers['Access-Control-Allow-Origin'] = '*'
#         rst.headers['Access-Control-Allow-Credentials'] = 'true'
#         rst.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
#         allow_headers = "Referer,Accept,Origin,User-Agent"
#         rst.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
#         return rst
#     return wrapper_fun



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/test',methods=['GET'])
# @allow_cross_domain
def test_interface():
    print("test connection ok")
    return jsonify({'status': 1,
                        'msg': '测试成功'})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    filename_prefix = file.filename.rsplit('.', 1)[0]
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        ir_image_path = os.path.join('./tmp/ct', filename_prefix.replace('ir', 'vi') + file.filename.rsplit('.', 1)[1])
        if os.path.exists(ir_image_path):
            return jsonify({'status': 0,
                        'msg': '尚未上传红外图像'})
        predict_image_path,predict_image_info,quality_image_info=core.main.fusion_and_detect(image_path,current_app.model)
        return jsonify({'status': 1,
                        'image_url': f'http://{host_ip}:9001/tmp/ct/' + file.filename,
                        'draw_url': str(os.path.join(f'http://{host_ip}:9001/',predict_image_path)),
                        'image_info': predict_image_info,
                        'quality_info': quality_image_info})

    return jsonify({'status': 0})


@app.route('/uploadir', methods=['GET', 'POST'])
# @allow_cross_domain
def upload_file2():
    file = request.files['file']
    filename_prefix = file.filename.rsplit('.', 1)[0]
    # print(filename_prefix)
    if not filename_prefix.endswith('_ir'):
        return jsonify({'status': 0,
                        'msg': '文件格式错误' + filename_prefix})
    
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        pid = core.main.c_main2(
            image_path, current_app.model, file.filename.rsplit('.', 1)[1])
        return jsonify({'status': 1,
                        'image_url': f'http://{host_ip}:9001/tmp/ct/' + pid
                        })

    return jsonify({'status': 0,
                    'msg': "error"})

@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
        
@app.route('/runs/<path:file>', methods=['GET'])
def show_photo_2(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'runs/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
    with app.app_context():
        current_app.model = FusionDetectot()
    # app.run(host='localhost', port=9001, debug=True)
    # docker 中的host ip要换成0.0.0.0表示监听来自任意IP地址的请求，即对外开放,不然localhost对应的ip只是docker内部的,而不能对应宿主机
    # host_ip="localhost"
    host_ip="127.0.0.1"
    app.run(host='0.0.0.0', port=9001,debug=False)
