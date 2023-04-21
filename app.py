from flask import Flask, url_for, redirect, request
from flask import render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
# import cv2
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, current_user, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'theusual'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(20), nullable=False)
    lastname = db.Column(db.String(20), nullable=False)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    firstname = StringField(validators=[
        InputRequired(), Length(min=3, max=20)], render_kw={"placeholder": "firstname"})
    lastname = StringField(validators=[
        InputRequired(), Length(min=3, max=20)], render_kw={"placeholder": "lastname"})

    submit = SubmitField('Register', render_kw={"class": "clearfix"})

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')
# model = load_model("./Resnet50/resnet50.h5")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = "./Yolov7/best.pt"
# Load the best weights from file
model = torch.hub.load("WongKinYiu/yolov7", "custom",
                       weights_path, trust_repo=True)
# model.to(device)

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((640, 640)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


@app.route("/")
def welcome():
    return render_template("home.html")


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('index'))
        else:
            form.username.errors.append('User not found')
    return render_template("login.html", form=form)


@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password,
                        firstname=form.firstname.data, lastname=form.lastname.data)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("register.html", form=form)


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/compare")
def compare():
    return render_template("comparator.html")


@app.route("/impact")
def impact():
    return render_template("impact.html")


@app.route("/upload_files", methods=['POST'])
def upload_files():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    print(file)
    if file.filename == '':
        return 'No selected file'

    image = Image.open(file.stream)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image)
    predictions = results.pandas().xyxy[0]['name'].tolist()

    # print(image)
    # # preprocess the image
    # image = np.array(image.resize((224, 224))) / 255.0
    # # make predictions
    # predictions = model.predict(np.array([image]))
    # # get the predicted class


# print the predicted class (assuming binary classification)
    # class_idx = np.argmax(predictions)
    return redirect(url_for('predict', class_idx=predictions[0]))


@app.route("/predict/<string:class_idx>")
def predict(class_idx):
    if class_idx == "Covid":
        prediction = "COVID"
    else:
        prediction = "NORMAL"
    return render_template('predict.html', prediction=prediction)

    # class_idx = request.args.get('class_idx')
    # if(class_idx == 0):
    #     return "Covid"
    # else:
    #     return "Normal"


if __name__ == "__main__":
    app.run(debug=True)
