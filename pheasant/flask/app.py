'''
参考
https://code.visualstudio.com/docs/python/tutorial-flask
'''

from flask import Flask
from flask import render_template
import re
from datetime import datetime
import locale

app = Flask(__name__)
locale.setlocale(locale.LC_TIME, 'ja_JP.UTF-8')

@app.route("/")
def home():
    # return "こんにちは, Flask♪"
    return render_template(
        "index.html"
    )

# @app.route("/pheasant/hello/<name>")
# def hello_there(name):
#     now = datetime.now()
#     formatted_now = now.strftime("%a曜日, %B %d日, %Y at %X")

#     #危険な文字はフィルタリングで弾かれる。
#     match_object = re.match("[a-zA-Z]+", name)

#     #今のところマルチバイト文字はマッチしない。
#     if match_object:
#         clean_name = match_object.group(0)
#     else:
#         clean_name = "どこかの誰か"

#     content = "お世話になっております, " + clean_name + "! 今は" + formatted_now
#     return content

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")
