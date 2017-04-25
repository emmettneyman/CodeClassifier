# Frontend for programming language classifier
# Built by Alex Matthys and Emmett Neyman

from flask import Flask, render_template, request, redirect, url_for, json, jsonify
import requests
import classifier

app = Flask(__name__)

@app.route('/')
def home():
    global classify
    classify = classifier.classifier(classifier.target_list, classifier.data_array)
    return render_template('index.html', code = "")

@app.route('/', methods = ['POST'])
def my_form_post():
    classify.predictCode(request.form['code'])
    result = classify.returnPrediction()
    return render_template('index.html', code = result)

def main():
    app.debug = True
    app.run()

if __name__ == '__main__':
    main()
