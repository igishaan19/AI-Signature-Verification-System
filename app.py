from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import shutil
import svm  
import preproc
import lineSweep

app = Flask(__name__)

app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
LINE_SWEEP_FOLDER = 'static/LineSweep_Results'
OCR_FOLDER = 'static/OCR_Results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(LINE_SWEEP_FOLDER, exist_ok=True)
os.makedirs(OCR_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/reload')
def reload_page():
    for folder in [app.config['UPLOAD_FOLDER'], LINE_SWEEP_FOLDER, OCR_FOLDER]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    return redirect('/')


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        for f in os.listdir(LINE_SWEEP_FOLDER):
            os.remove(os.path.join(LINE_SWEEP_FOLDER, f))

        test_dest = os.path.join(LINE_SWEEP_FOLDER, filename)
        shutil.copy(save_path, test_dest)

        flash('Image successfully uploaded')
        return render_template('home.html', filename=filename)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/process_ocr', methods=['POST'])
def process_ocr():

    result = svm.svm_algo()  

    if result == "Genuine":
        final = "Genuine Signature"
    else:
        final = "Forged Signature"

    return render_template("home.html", result=final)


if __name__ == '__main__':
    app.run(debug=True)
