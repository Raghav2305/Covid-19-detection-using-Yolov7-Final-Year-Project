from flask import Flask, url_for, redirect
from flask import render_template

app = Flask(__name__, template_folder='templates')


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/compare")
def compare():
    return render_template("comparator.html")


@app.route("/impact")
def impact():
    return render_template("impact.html")


if __name__ == "__main__":
    app.run(debug=True)
