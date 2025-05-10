from flask import Flask, jsonify, request

from src.classifier import classify_file

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/classify_file", methods=["POST"])
def classify_file_route():

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed: {file.filename}"}), 400

    file_class = classify_file(file)
    return jsonify({"file_class": file_class}), 200


if __name__ == "__main__":
    app.run(debug=True)
