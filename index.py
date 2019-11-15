from flask import *
import pandas as pd
import requests, json

app = Flask(__name__)

@app.route("/")
def home():
	return render_template('home.html')

if __name__=='__main__':
	app.run()
