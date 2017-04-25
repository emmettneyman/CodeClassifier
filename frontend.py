# Frontend for programming language classifier
# Built by Alex Matthys and Emmett Neyman

from flask import Flask, render_template, request
import requests
import classifier as cl

app = Flask(__name__)


@app.route('/')
def home():
    global classify
    classify = cl.classifier(cl.target_list, cl.data_array)
    return render_template('index.html', code="")


@app.route('/', methods=['POST'])
def my_form_post():
    global classify
    classify.predictCode(request.form['code'])
    result = classify.returnPrediction()
    return render_template('index.html', code=result)


def main():
    app.debug = True
    app.run()


if __name__ == '__main__':
    main()
