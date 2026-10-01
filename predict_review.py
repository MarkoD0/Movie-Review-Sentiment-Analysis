# predict_review.py
#
# Samostalna GUI aplikacija za predvidjanje sentimenta filmskih recenzija, koja koristi modele obucene u okviru projekta imdb_sentiment thesis.
# Pokrece se iz korena projekta preko python predict_review.py (psotoji i .exe aplikacija za lakse pokretanje)
#
# Glavno predvidjanje koristi Logisticku regresiju (baseline) - najbolji model utvrdjen u 10_evaluation.ipynb (statisticki izjednacen sa SVM baseline modelom, ali jasno bolji od svih ostalih modela).
# Panel sa rezultatima dodatno prikazuje kako svaki DRUGI obuceni model ocenjuje istu recenziju, kroz sve 4 varijante pretprocesiranja (ukupno 24 reda).
# Teski modeli Random Forest i KNN namerno su izostavljeni (umesto njih koriste se njihove lite verzije) zbog velike velicine fajlova (do priblizno 6,4 GB, odnosno 1,5 GB po varijanti).

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import joblib

from src.preprocessing import (
    init_preprocessing,
    basic_clean,
    baseline_tokens,
    stem_tokens,
    lemmatize_tokens,
    lancaster_stem_tokens,
)


# ---------------------------------------------------------------------------
# Konfiguracija
# ---------------------------------------------------------------------------
#
# Namerno izostavljeni "random_forest" i "knn" (heavy verzije) - koriste se njihove "lite" verzije umesto njih.
# Redosled prati F1 rangiranje iz 10_evaluation.ipynb (najbolji ka najlosijem).
#
# MODELS_DIR se racuna u odnosu na lokaciju bundle-ovanih podataka (ako je exe napravljen sa --add-data "models;models", oni zive unutar sys._MEIPASS), a u suprotnom u odnosu na lokaciju skripte/exe-a (spoljasnji models/ folder pored exe-a).

if getattr(sys, "frozen", False):
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))    # Pokrenuto kao PyInstaller .exe. sys._MEIPASS postoji cim PyInstaller upakuje bilo kakve data fajlove (i za --onefile i za --onedir) i pokazuje na mesto gde se ti fajlovi nalaze pri izvrsavanju.
else:
    BASE_DIR = Path(__file__).resolve().parent      # Pokrenuto kao obican .py fajl

MODELS_DIR = BASE_DIR / "models"

MODEL_FAMILIES = ["logistic_regression", "svm", "naive_bayes", "lite_random_forest", "lite_knn", "decision_tree"]
VARIANTS = ["base", "stem", "lem", "lanc"]

MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "svm": "SVM",
    "lite_random_forest": "Random Forest (lite)",
    "lite_knn": "KNN (lite)",
}
VARIANT_DISPLAY_NAMES = {
    "base": "Baseline",
    "stem": "Stemming",
    "lem": "Lemmatization",
    "lanc": "Lancaster",
}

MAIN_MODEL_FAMILY = "logistic_regression"
MAIN_MODEL_VARIANT = "base"

LABEL_MAP = {0: "Negative", 1: "Positive"}


# ---------------------------------------------------------------------------
# Ucitavanje modela i vektorizatora
# ---------------------------------------------------------------------------

# Ucitava 4 fitovana TF-IDF vektorizatora, po jedan za svaku preprocessing varijantu.
def load_vectorizers() -> dict:
    vectorizers = {}
    for variant in VARIANTS:
        path = MODELS_DIR / f"tfidf_vectorizer_{variant}.joblib"
        try:
            vectorizers[variant] = joblib.load(path)
        except FileNotFoundError:
            vectorizers[variant] = None
            print(f"WARNING: missing vectorizer file: {path}")
    return vectorizers

# Ucitava TruncatedSVD reducer-e specificne za lite_knn (svd_reducer_lite_knn_<variant>.joblib).
# Ostali modeli rade direktno nad TF-IDF matricom i ne trebaju reducer.
def load_svd_reducers() -> dict:
    reducers = {}
    for variant in VARIANTS:
        path = MODELS_DIR / f"svd_reducer_lite_knn_{variant}.joblib"
        try:
            reducers[variant] = joblib.load(path)
        except FileNotFoundError:
            reducers[variant] = None
            print(f"WARNING: missing SVD reducer file: {path}")
    return reducers

# Ucitava svaku kombinaciju model x varijanta (bez heavy random_forest/knn).
# Vraca recnik: (model_family, variant) -> fitovan model (ili None ako fajl nedostaje).
def load_models() -> dict:
    models = {}
    for model_family in MODEL_FAMILIES:
        for variant in VARIANTS:
            path = MODELS_DIR / f"{model_family}_{variant}.joblib"
            try:
                models[(model_family, variant)] = joblib.load(path)
            except FileNotFoundError:
                models[(model_family, variant)] = None
                print(f"WARNING: missing model file: {path}")
    return models


# ---------------------------------------------------------------------------
# Predikcija
# ---------------------------------------------------------------------------
#
# Preprocessing se radi jednom, vektorizacija jednom po varijanti (ne po modelu - svi modeli iste varijante dele isti TF-IDF vektor).
# lite_knn je poseban slucaj: trenirana je nad SVD-redukovanim feature prostorom (videti 8.2.x notebook-ove), pa se za nju TF-IDF vektor mora dodatno provuci kroz sacuvan svd_reducer_lite_knn_<variant>.joblib pre poziva .predict(). Ostali modeli rade direktno nad TF-IDF matricom.
# Vraca:
#   main_prediction  - "Positive" / "Negative" / None (ako glavni model nedostaje)
#   main_confidence  - verovatnoca predikcije glavnog modela (0-1) ili None ako nije dostupna
#   breakdown        - lista (model_family, variant, prediction_label) za svih 24 kombinacije

def predict_all(review_text: str, vectorizers: dict, svd_reducers: dict, models: dict):
    # basic_clean() se poziva jednom, daje listu tokena
    # Svaka grana (baseline/stem/lem/lanc) zatim samo spaja te iste tokene na svoj nacin - identicno kao u 2_preprocessing.ipynb
    tokens = basic_clean(review_text)
    variant_texts = {
        "base": baseline_tokens(tokens),
        "stem": stem_tokens(tokens),
        "lem": lemmatize_tokens(tokens),
        "lanc": lancaster_stem_tokens(tokens),
    }

    # Vektorizacija - jednom po varijanti, deli se izmedju svih modela te varijante
    variant_features = {}
    for variant in VARIANTS:
        vectorizer = vectorizers.get(variant)
        if vectorizer is None:
            variant_features[variant] = None
            continue
        variant_features[variant] = vectorizer.transform([variant_texts[variant]])

    # lite_knn - dodatna SVD redukcija nad TF-IDF vektorom, jednom po varijanti (float16 posle transform-a, identicno kao u 8.2.x notebook-ovima)
    lite_knn_features = {}
    for variant in VARIANTS:
        reducer = svd_reducers.get(variant)
        tfidf_vector = variant_features.get(variant)
        if reducer is None or tfidf_vector is None:
            lite_knn_features[variant] = None
            continue
        lite_knn_features[variant] = reducer.transform(tfidf_vector).astype(np.float16)

    main_prediction = None
    main_confidence = None
    breakdown = []
    for model_family in MODEL_FAMILIES:
        for variant in VARIANTS:
            model = models.get((model_family, variant))
            features = lite_knn_features.get(variant) if model_family == "lite_knn" else variant_features.get(variant)

            if model is None or features is None:
                label = "N/A (missing file)"
            else:
                pred = model.predict(features)[0]
                label = LABEL_MAP[int(pred)]

            breakdown.append((model_family, variant, label))

            if model_family == MAIN_MODEL_FAMILY and variant == MAIN_MODEL_VARIANT:
                main_prediction = label
                if model is not None and features is not None and hasattr(model, "predict_proba"):
                    # Verovatnoca dodeljena predvidjenoj klasi - "koliko je model siguran u svoj odgovor"
                    proba = model.predict_proba(features)[0]
                    main_confidence = float(proba[int(pred)])

    return main_prediction, main_confidence, breakdown


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SentimentApp(tk.Tk):
    def __init__(self, vectorizers: dict, svd_reducers: dict, models: dict):
        super().__init__()
        self.vectorizers = vectorizers
        self.svd_reducers = svd_reducers
        self.models = models

        self.title("IMDB Movie Review Sentiment Analyzer")
        self.geometry("700x650")
        self.minsize(600, 550)

        self._build_widgets()

    def _build_widgets(self):
        padding = {"padx": 12, "pady": 8}

        input_label = ttk.Label(self, text="Enter a movie review:", font=("Segoe UI", 11, "bold"))
        input_label.pack(anchor="w", **padding)

        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=False, padx=12)

        self.review_text = tk.Text(text_frame, height=8, wrap="word", font=("Segoe UI", 10))
        self.review_text.pack(side="left", fill="both", expand=True)

        text_scroll = ttk.Scrollbar(text_frame, command=self.review_text.yview)
        text_scroll.pack(side="right", fill="y")
        self.review_text.configure(yscrollcommand=text_scroll.set)

        self.analyze_button = ttk.Button(self, text="Analyze Review", command=self.on_analyze)
        self.analyze_button.pack(pady=10)

        self.result_label = ttk.Label(self, text="", font=("Segoe UI", 18, "bold"))
        self.result_label.pack(pady=(0, 2))

        self.confidence_label = ttk.Label(self, text="", font=("Segoe UI", 10))
        self.confidence_label.pack(pady=(0, 2))

        self.result_sublabel = ttk.Label(self, text="", font=("Segoe UI", 9))
        self.result_sublabel.pack(pady=(0, 10))

        breakdown_label = ttk.Label(self, text="Prediction by model / preprocessing variant (sorted from most to least reliable):", font=("Segoe UI", 10, "bold"),)
        breakdown_label.pack(anchor="w", padx=12)

        # Matrica model x preprocessing varijanta - Frame + grid od Label widget-a (svaka celija (model, varijanta) ima svoju nezavisnu Positive/Negative boju).
        matrix_container = ttk.Frame(self)
        matrix_container.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        canvas = tk.Canvas(matrix_container, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        matrix_scroll = ttk.Scrollbar(matrix_container, orient="vertical", command=canvas.yview)
        matrix_scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=matrix_scroll.set)

        self.matrix_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.matrix_frame, anchor="nw")
        self.matrix_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self._build_matrix_header()

    def _build_matrix_header(self):
        header_font = ("Segoe UI", 9, "bold")
        ttk.Label(self.matrix_frame, text="Model", font=header_font, width=20).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        for col, variant in enumerate(VARIANTS, start=1):
            ttk.Label(self.matrix_frame, text=VARIANT_DISPLAY_NAMES[variant], font=header_font, width=13, anchor="center").grid(row=0, column=col, padx=4, pady=4)

    def on_analyze(self):
        review_text = self.review_text.get("1.0", "end").strip()
        if not review_text:
            messagebox.showwarning("Empty review", "Please enter a movie review before analyzing.")
            return

        self.analyze_button.configure(state="disabled")
        self.result_label.configure(text="Analyzing...", foreground="black")
        self.confidence_label.configure(text="")
        self.result_sublabel.configure(text="")
        self.update_idletasks()

        try:
            main_prediction, main_confidence, breakdown = predict_all(review_text, self.vectorizers, self.svd_reducers, self.models)
        except Exception as exc:
            messagebox.showerror("Prediction failed", f"An error occurred while analyzing the review:\n\n{exc}")
            self.result_label.configure(text="")
            self.analyze_button.configure(state="normal")
            return

        self._display_results(main_prediction, main_confidence, breakdown)
        self.analyze_button.configure(state="normal")

    def _display_results(self, main_prediction, main_confidence, breakdown):
        if main_prediction is None:
            self.result_label.configure(text="Main model unavailable", foreground="#999999")
            self.confidence_label.configure(text="")
            self.result_sublabel.configure(text=f"Missing file: {MAIN_MODEL_FAMILY}_{MAIN_MODEL_VARIANT}.joblib")
        else:
            color = "#1a7a33" if main_prediction == "Positive" else "#b32020"
            self.result_label.configure(text=main_prediction.upper(), foreground=color)
            if main_confidence is not None:
                self.confidence_label.configure(text=f"{main_confidence * 100:.1f}% confidence")
            else:
                self.confidence_label.configure(text="")
            self.result_sublabel.configure(text=f"(Predicted by {MODEL_DISPLAY_NAMES[MAIN_MODEL_FAMILY]}, {VARIANT_DISPLAY_NAMES[MAIN_MODEL_VARIANT]} - best model per the thesis evaluation)")

        # breakdown je flat lista (model_family, variant, label) - pregrupisemo po modelu radi matricnog prikaza
        by_model = {model_family: {} for model_family in MODEL_FAMILIES}
        for model_family, variant, label in breakdown:
            by_model[model_family][variant] = label

        # Brisemo prethodne redove matrice (sve osim zaglavlja, koje je uvek row 0)
        for widget in self.matrix_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 0:
                widget.destroy()

        for row, model_family in enumerate(MODEL_FAMILIES, start=1):
            ttk.Label(self.matrix_frame, text=MODEL_DISPLAY_NAMES[model_family], font=("Segoe UI", 9), width=20).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            for col, variant in enumerate(VARIANTS, start=1):
                label = by_model[model_family].get(variant, "N/A (missing file)")
                if label.startswith("N/A"):
                    text, color = "N/A", "#999999"
                elif label == "Positive":
                    text, color = "Positive", "#1a7a33"
                else:
                    text, color = "Negative", "#b32020"

                ttk.Label(self.matrix_frame, text=text, foreground=color, font=("Segoe UI", 9), width=13, anchor="center").grid(row=row, column=col, padx=4, pady=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_preprocessing()

    print("Loading vectorizers...")
    vectorizers = load_vectorizers()

    print("Loading SVD reducers (lite_knn)...")
    svd_reducers = load_svd_reducers()

    print("Loading models...")
    models = load_models()

    if vectorizers.get(MAIN_MODEL_VARIANT) is None or models.get((MAIN_MODEL_FAMILY, MAIN_MODEL_VARIANT)) is None:
        print("WARNING: the main model (Logistic Regression, baseline) or its vectorizer could not be loaded.")

    app = SentimentApp(vectorizers, svd_reducers, models)
    app.mainloop()


if __name__ == "__main__":
    main()
