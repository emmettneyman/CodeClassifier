# Frontend for programming language classifier
# Built by Alex Matthys and Emmett Neyman

from flask import Flask, render_template, request, redirect, url_for, json, jsonify
import requests
import logging

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', code = "")

@app.route('/', methods = ['POST'])
def my_form_post():
    code = request.form['code']
    return render_template('index.html', code = code)

def main():
    app.debug = True
    app.run()

if __name__ == '__main__':
    main()
