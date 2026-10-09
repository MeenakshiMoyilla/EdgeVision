from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    session,
    send_from_directory
)

import os
import uuid
import cv2
import time
import sqlite3

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from edge_detection import (
    preprocess_image,
    sobel_edge_detection,
    prewitt_edge_detection,
    canny_edge_detection
)


app = Flask(__name__)

# Secret key for login sessions
app.secret_key = "edgevision-development-secret-key"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
DATABASE = "database.db"

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_db_connection():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()


init_database()


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def login_required(function):

    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )

        return function(*args, **kwargs)

    wrapper.__name__ = function.__name__

    return wrapper


# --------------------------------------------------
# Landing Page
# --------------------------------------------------

@app.route("/")
def landing():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "landing.html"
    )


# --------------------------------------------------
# Register
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )


    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # Validate name

        if not name:

            return render_template(
                "register.html",
                error="Please enter your name.",
                name=name,
                email=email
            )


        # Validate email

        if not email or "@" not in email:

            return render_template(
                "register.html",
                error="Please enter a valid email address.",
                name=name,
                email=email
            )


        # Validate password

        if len(password) < 6:

            return render_template(
                "register.html",
                error="Password must contain at least 6 characters.",
                name=name,
                email=email
            )


        # Confirm password

        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match.",
                name=name,
                email=email
            )


        connection = get_db_connection()


        # Check existing account

        existing_user = connection.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


        if existing_user:

            connection.close()

            return render_template(
                "register.html",
                error="An account with this email already exists.",
                name=name,
                email=email
            )


        # Hash password

        hashed_password = generate_password_hash(
            password
        )


        # Create user

        connection.execute(
            """
            INSERT INTO users
            (name, email, password)
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                hashed_password
            )
        )

        connection.commit()

        connection.close()


        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# --------------------------------------------------
# Login
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )


    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        connection = get_db_connection()


        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


        connection.close()


        if user is None:

            return render_template(
                "login.html",
                error="Invalid email or password.",
                email=email
            )


        if not check_password_hash(
            user["password"],
            password
        ):

            return render_template(
                "login.html",
                error="Invalid email or password.",
                email=email
            )


        # Create login session

        session["user_id"] = user["id"]

        session["user_name"] = user["name"]

        session["user_email"] = user["email"]


        return redirect(
            url_for("dashboard")
        )


    return render_template(
        "login.html"
    )


# --------------------------------------------------
# Logout
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("landing")
    )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user_name=session.get(
            "user_name"
        )
    )


# --------------------------------------------------
# Image Analysis Page
# --------------------------------------------------

@app.route("/analyze")
@login_required
def analyze():

    return render_template(
        "dashboard.html",
        user_name=session.get(
            "user_name"
        )
    )


# --------------------------------------------------
# Upload and Process Image
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
@login_required
def upload_image():

    if "image" not in request.files:

        return render_template(
            "dashboard.html",
            error="Please select an image.",
            user_name=session.get(
                "user_name"
            )
        )


    file = request.files["image"]


    if file.filename == "":

        return render_template(
            "dashboard.html",
            error="Please select an image.",
            user_name=session.get(
                "user_name"
            )
        )


    if not allowed_file(file.filename):

        return render_template(
            "dashboard.html",
            error="Invalid file type. Please upload JPG, JPEG or PNG.",
            user_name=session.get(
                "user_name"
            )
        )


    # --------------------------------------------------
    # Canny Thresholds
    # --------------------------------------------------

    try:

        threshold1 = int(
            request.form.get(
                "threshold1",
                50
            )
        )

        threshold2 = int(
            request.form.get(
                "threshold2",
                150
            )
        )

    except ValueError:

        return render_template(
            "dashboard.html",
            error="Canny thresholds must be valid numbers.",
            user_name=session.get(
                "user_name"
            )
        )


    if threshold1 < 0 or threshold1 > 255:

        return render_template(
            "dashboard.html",
            error="Lower threshold must be between 0 and 255.",
            user_name=session.get(
                "user_name"
            )
        )


    if threshold2 < 0 or threshold2 > 255:

        return render_template(
            "dashboard.html",
            error="Upper threshold must be between 0 and 255.",
            user_name=session.get(
                "user_name"
            )
        )


    if threshold1 >= threshold2:

        return render_template(
            "dashboard.html",
            error="Lower threshold must be smaller than upper threshold.",
            user_name=session.get(
                "user_name"
            )
        )


    # --------------------------------------------------
    # Save Image
    # --------------------------------------------------

    original_filename = secure_filename(
        file.filename
    )

    filename = (
        str(uuid.uuid4())
        + "_"
        + original_filename
    )


    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(image_path)


    try:

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        image, gray, blurred = preprocess_image(
            image_path
        )


        # --------------------------------------------------
        # Sobel
        # --------------------------------------------------

        start_time = time.perf_counter()

        sobel = sobel_edge_detection(
            blurred
        )

        sobel_time = (
            time.perf_counter()
            - start_time
        )


        # --------------------------------------------------
        # Prewitt
        # --------------------------------------------------

        start_time = time.perf_counter()

        prewitt = prewitt_edge_detection(
            blurred
        )

        prewitt_time = (
            time.perf_counter()
            - start_time
        )


        # --------------------------------------------------
        # Canny
        # --------------------------------------------------

        start_time = time.perf_counter()

        canny = canny_edge_detection(
            blurred,
            threshold1,
            threshold2
        )

        canny_time = (
            time.perf_counter()
            - start_time
        )


        # --------------------------------------------------
        # Output filenames
        # --------------------------------------------------

        sobel_filename = (
            "sobel_" + filename
        )

        prewitt_filename = (
            "prewitt_" + filename
        )

        canny_filename = (
            "canny_" + filename
        )


        # --------------------------------------------------
        # Output paths
        # --------------------------------------------------

        sobel_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            sobel_filename
        )

        prewitt_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            prewitt_filename
        )

        canny_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            canny_filename
        )


        # --------------------------------------------------
        # Save results
        # --------------------------------------------------

        cv2.imwrite(
            sobel_path,
            sobel
        )

        cv2.imwrite(
            prewitt_path,
            prewitt
        )

        cv2.imwrite(
            canny_path,
            canny
        )


        # --------------------------------------------------
        # Image information
        # --------------------------------------------------

        height, width = gray.shape

        file_size = (
            os.path.getsize(image_path)
            / (1024 * 1024)
        )


        total_time = (
            sobel_time
            + prewitt_time
            + canny_time
        )


        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        return render_template(
            "dashboard.html",

            user_name=session.get(
                "user_name"
            ),

            original_filename=filename,

            sobel_filename=sobel_filename,

            prewitt_filename=prewitt_filename,

            canny_filename=canny_filename,

            threshold1=threshold1,

            threshold2=threshold2,

            image_width=width,

            image_height=height,

            file_size=round(
                file_size,
                2
            ),

            sobel_time=round(
                sobel_time * 1000,
                3
            ),

            prewitt_time=round(
                prewitt_time * 1000,
                3
            ),

            canny_time=round(
                canny_time * 1000,
                3
            ),

            total_time=round(
                total_time * 1000,
                3
            )
        )


    except Exception as e:

        return render_template(
            "dashboard.html",
            error=f"Error processing image: {str(e)}",
            user_name=session.get(
                "user_name"
            )
        )


# --------------------------------------------------
# File Too Large
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "dashboard.html",
        error="File is too large. Maximum allowed size is 10 MB.",
        user_name=session.get(
            "user_name"
        )
    ), 413


# --------------------------------------------------
# Serve Uploaded Images
# --------------------------------------------------

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# --------------------------------------------------
# Serve Output Images
# --------------------------------------------------

@app.route("/outputs/<filename>")
@login_required
def output_file(filename):

    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )