import time

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    count = 3
    return 'Hello {} times.\n'.format(count)
