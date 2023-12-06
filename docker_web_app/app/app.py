import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
import backend

from flask import Flask



app = Flask(__name__)

if not os.path.exists("uploads/"):
	os.mkdir("uploads")
app.config["UPLOAD_FOLDER"] = 'uploads/'


@app.route("/")
def index():
	return render_template('index.html')

@app.route("/", methods=["POST"])
def upload_vide():
	# check that POST request has the file
	if "file" not in request.files:
		flash("No file")
		return redirect(request.url)
	file = request.files["file"]
	if file.filename == "":
		flash("No video selected")
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
		return render_template("video.html", filename=filename)

@app.route("/video/<filename>")
def video(filename):
	return Response(backend.generate_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__": 
    app.run()