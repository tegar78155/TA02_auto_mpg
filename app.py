import logging
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, static_folder="public", static_url_path="", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-secret-key")
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_regresi_auto_mpg.joblib")
MODEL = None
MODEL_ERROR = None

try:
    MODEL = joblib.load(MODEL_PATH)
    logger.info("Model berhasil dimuat dari %s", MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    MODEL_ERROR = str(exc)
    logger.exception("Gagal memuat model")


def _validate_weight(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None, "Masukkan berat kendaraan terlebih dahulu."
    try:
        value = float(raw)
    except ValueError:
        return None, "Masukkan berat kendaraan dalam angka yang valid."
    if value < 1000 or value > 6000:
        return None, "Masukkan berat kendaraan antara 1.000 dan 6.000 pound."
    return value, None


@app.route("/", methods=["GET", "POST"])
def index():
    prediksi = None
    berat = ""
    error = None

    if request.method == "POST":
        if MODEL is None:
            error = "Model tidak dapat dimuat. Hubungi administrator."
        else:
            berat = request.form.get("weight", "")
            nilai_berat, err = _validate_weight(berat)
            if err:
                error = err
            else:
                try:
                    data_input = pd.DataFrame({"weight": [nilai_berat]})
                    hasil = MODEL.predict(data_input)[0]
                    prediksi = round(float(hasil), 2)
                    berat = str(nilai_berat)
                except Exception:  # noqa: BLE001
                    logger.exception("Kesalahan saat prediksi")
                    error = "Terjadi kesalahan saat memproses prediksi."

    return render_template("index.html", prediksi=prediksi, berat=berat, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=app.config["DEBUG"])
