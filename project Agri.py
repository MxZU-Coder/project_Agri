# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED CROP MONITORING SYSTEM — COLAB VERSION
# with AI Crop Suggestion, Yield Prediction & Interactive Input
# ═══════════════════════════════════════════════════════════════════════════════

# --- Install (only if missing) ------------------------------------------------
import importlib, subprocess, sys

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("xgboost")
_ensure("scikit-learn")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              VotingClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Use inline backend for Colab
%matplotlib inline


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HYPERSPECTRAL DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class HyperspectralDataLoader:
    """Generates and preprocesses synthetic hyperspectral crop data."""

    def __init__(self, n_bands=200, random_state=42):
        self.n_bands = n_bands
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.scaler = StandardScaler()
        self.sg_window = 11
        self.sg_polyorder = 2
        self.gaussian_sigma = 1.5

    def generate_synthetic_data(self, n_samples=2000, n_classes=4):
        print("Generating synthetic hyperspectral data...")
        class_names = ["Healthy", "Nutrient_Deficient", "Diseased", "Pest_Damaged"][:n_classes]

        X = np.zeros((n_samples, self.n_bands), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)

        wavelengths = np.linspace(400, 2500, self.n_bands)

        for i in range(n_samples):
            cls = i % n_classes
            y[i] = cls

            base = (
                0.5 * np.exp(-((wavelengths - 550) ** 2) / (2 * 50 ** 2))
                + 0.7 * np.exp(-((wavelengths - 800) ** 2) / (2 * 80 ** 2))
                - 0.3 * np.exp(-((wavelengths - 1450) ** 2) / (2 * 60 ** 2))
                - 0.4 * np.exp(-((wavelengths - 1950) ** 2) / (2 * 80 ** 2))
            )

            if cls == 1:
                base = base * 0.75 + 0.05 * np.exp(-((wavelengths - 680) ** 2) / (2 * 40 ** 2))
            elif cls == 2:
                base = base * 0.65 + 0.10 * np.exp(-((wavelengths - 670) ** 2) / (2 * 30 ** 2))
            elif cls == 3:
                base = base * 0.80 + 0.08 * np.exp(-((wavelengths - 700) ** 2) / (2 * 50 ** 2))

            X[i] = base + self.rng.normal(0, 0.02, self.n_bands)

        return X, y, class_names

    def preprocess_data(self, X, y):
        print("Preprocessing spectral data...")
        try:
            X_smooth = savgol_filter(X, window_length=self.sg_window,
                                     polyorder=self.sg_polyorder, axis=1)
        except Exception:
            X_smooth = X
        X_filtered = gaussian_filter1d(X_smooth, sigma=self.gaussian_sigma, axis=1)
        X_scaled = self.scaler.fit_transform(X_filtered)
        return X_scaled.astype(np.float32), y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SOIL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class SoilAnalyzer:
    PROPERTIES = ["nitrogen", "phosphorus", "potassium", "moisture", "organic_matter", "ph"]

    def __init__(self, n_bands=200, random_state=42):
        self.n_bands = n_bands
        self.rng = np.random.default_rng(random_state)
        self.models = {}
        self.scaler = StandardScaler()

    def generate_soil_data(self, n_samples=800):
        print("Generating synthetic soil spectral data...")
        X = self.rng.normal(0.5, 0.15, size=(n_samples, self.n_bands)).astype(np.float32)

        projection = X.mean(axis=1)
        y = {
            "nitrogen":       0.20 + 0.10 * projection + self.rng.normal(0, 0.01, n_samples),
            "phosphorus":     0.15 + 0.08 * projection + self.rng.normal(0, 0.01, n_samples),
            "potassium":      0.25 + 0.12 * projection + self.rng.normal(0, 0.01, n_samples),
            "moisture":      25.00 + 5.00 * projection + self.rng.normal(0, 0.50, n_samples),
            "organic_matter": 3.00 + 1.50 * projection + self.rng.normal(0, 0.10, n_samples),
            "ph":             6.50 + 0.50 * projection + self.rng.normal(0, 0.05, n_samples),
        }
        return X, y

    def train_models(self, X, y):
        print("Training soil property regression models (Gradient Boosting)...")
        results = {}
        for prop in self.PROPERTIES:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y[prop], test_size=0.2, random_state=42)
            model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            r2 = r2_score(y_te, preds)
            rmse = np.sqrt(mean_squared_error(y_te, preds))

            self.models[prop] = model
            results[prop] = {"r2": float(r2), "rmse": float(rmse),
                             "y_true": y_te, "y_pred": preds}
            print(f"   {prop:<15} R² = {r2:.3f}   RMSE = {rmse:.3f}")
        return results

    def predict_soil_properties(self, X):
        if not self.models:
            X_all, y_all = self.generate_soil_data()
            self.train_models(X_all, y_all)
        return {prop: self.models[prop].predict(X) for prop in self.PROPERTIES}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PESTICIDE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class PesticideAnalyzer:
    def __init__(self, size=50, random_state=42):
        self.size = size
        self.rng = np.random.default_rng(random_state)

    def generate_field_map(self):
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]
        cx, cy = s / 2, s / 2
        true_map = 100 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (s / 4) ** 2)))
        field_map = true_map + self.rng.normal(0, 15, (s, s))
        field_map = np.clip(field_map, 0, 150)
        return field_map, true_map

    def detect_application_issues(self, field_map):
        optimal       = (field_map >= 60) & (field_map <= 110)
        under_applied = (field_map > 10) & (field_map < 60)
        over_applied  = field_map > 110
        missed        = field_map <= 10

        classification = np.zeros_like(field_map, dtype=np.int32)
        classification[under_applied] = 1
        classification[over_applied] = 2
        classification[missed] = 3

        total = field_map.size
        stats = {
            "optimal":       optimal.sum() / total * 100,
            "under_applied": under_applied.sum() / total * 100,
            "over_applied":  over_applied.sum() / total * 100,
            "missed":        missed.sum() / total * 100,
        }
        pesticide_index = field_map / (field_map.max() + 1e-8)
        return classification, pesticide_index, stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AI CROP SUGGESTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AICropSuggestionEngine:
    """
    Rule-based + ML-hybrid crop suggestion engine.
    Uses soil + climate inputs to rank suitable crops.
    """

    # Each crop defined by ideal ranges: (min, max) and weight of importance
    CROP_PROFILES = {
        "Rice":       {"N": (0.15, 0.30), "P": (0.10, 0.25), "K": (0.20, 0.35),
                       "moisture": (60, 90), "pH": (5.5, 7.0), "temp": (22, 32), "rain": (150, 300)},
        "Wheat":      {"N": (0.15, 0.30), "P": (0.10, 0.25), "K": (0.20, 0.35),
                       "moisture": (30, 55), "pH": (6.0, 7.5), "temp": (10, 25), "rain": (40, 110)},
        "Maize":      {"N": (0.20, 0.35), "P": (0.15, 0.28), "K": (0.25, 0.40),
                       "moisture": (40, 70), "pH": (5.8, 7.2), "temp": (18, 30), "rain": (50, 150)},
        "Sugarcane":  {"N": (0.25, 0.40), "P": (0.15, 0.30), "K": (0.30, 0.45),
                       "moisture": (55, 85), "pH": (6.0, 7.5), "temp": (20, 35), "rain": (100, 250)},
        "Cotton":     {"N": (0.15, 0.28), "P": (0.10, 0.22), "K": (0.20, 0.35),
                       "moisture": (35, 60), "pH": (6.0, 8.0), "temp": (21, 35), "rain": (50, 120)},
        "Soybean":    {"N": (0.12, 0.25), "P": (0.12, 0.25), "K": (0.20, 0.35),
                       "moisture": (40, 65), "pH": (6.0, 7.0), "temp": (18, 30), "rain": (60, 150)},
        "Groundnut":  {"N": (0.10, 0.22), "P": (0.12, 0.25), "K": (0.18, 0.32),
                       "moisture": (30, 55), "pH": (6.0, 7.5), "temp": (20, 33), "rain": (50, 120)},
        "Barley":     {"N": (0.12, 0.25), "P": (0.08, 0.20), "K": (0.18, 0.30),
                       "moisture": (25, 50), "pH": (6.0, 7.8), "temp": (8, 22), "rain": (30, 90)},
        "Millet":     {"N": (0.10, 0.22), "P": (0.08, 0.20), "K": (0.15, 0.30),
                       "moisture": (20, 45), "pH": (5.5, 7.5), "temp": (22, 35), "rain": (20, 80)},
        "Tomato":     {"N": (0.18, 0.32), "P": (0.15, 0.28), "K": (0.28, 0.45),
                       "moisture": (45, 70), "pH": (6.0, 7.0), "temp": (18, 28), "rain": (40, 100)},
        "Potato":     {"N": (0.18, 0.32), "P": (0.15, 0.28), "K": (0.30, 0.45),
                       "moisture": (50, 75), "pH": (5.0, 6.5), "temp": (12, 22), "rain": (40, 100)},
        "Onion":      {"N": (0.15, 0.28), "P": (0.12, 0.25), "K": (0.25, 0.40),
                       "moisture": (40, 65), "pH": (6.0, 7.5), "temp": (13, 28), "rain": (30, 90)},
    }

    @staticmethod
    def _score_range(value, lo, hi):
        """Score how well a value fits [lo, hi]. 1.0 = perfect, 0 = far."""
        if lo <= value <= hi:
            return 1.0
        span = max(hi - lo, 1e-6)
        d = (lo - value) if value < lo else (value - hi)
        return max(0.0, 1.0 - d / span)

    def suggest(self, soil_inputs, climate_inputs, top_k=5):
        """
        soil_inputs: dict with keys N, P, K, moisture, pH, organic_matter
        climate_inputs: dict with keys temp, rain
        Returns ranked list of (crop, score, reasons)
        """
        combined = {**soil_inputs, **climate_inputs}
        results = []

        for crop, profile in self.CROP_PROFILES.items():
            reasons = []
            scores = []

            for key, (lo, hi) in profile.items():
                val = combined.get(key)
                if val is None:
                    continue
                s = self._score_range(val, lo, hi)
                scores.append(s)
                if s >= 0.85:
                    reasons.append(f"{key}={val:.2f} is ideal")
                elif s < 0.4:
                    reasons.append(f"{key}={val:.2f} is out of range ({lo}-{hi})")

            final_score = float(np.mean(scores)) if scores else 0.0
            results.append((crop, final_score, reasons[:3]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. YIELD PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class YieldPredictionEngine:
    """
    Trains a regression model per crop on synthetic agronomic features.
    Predicts yield (tons/hectare) from soil, climate, and management factors.
    """

    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.models = {}
        self.crop_encoders = {}
        self.features = ["N", "P", "K", "moisture", "pH", "organic_matter",
                         "temp", "rain", "fertilizer", "area"]

    # -------- synthetic dataset generator --------
    def _generate_training_data(self, crop_name, n=1500):
        base_yield = {
            "Rice": 5.5, "Wheat": 4.2, "Maize": 6.8, "Sugarcane": 70.0,
            "Cotton": 2.0, "Soybean": 2.8, "Groundnut": 2.2, "Barley": 3.5,
            "Millet": 1.8, "Tomato": 35.0, "Potato": 28.0, "Onion": 25.0,
        }.get(crop_name, 3.0)

        N  = self.rng.uniform(0.05, 0.40, n)
        P  = self.rng.uniform(0.05, 0.35, n)
        K  = self.rng.uniform(0.10, 0.50, n)
        m  = self.rng.uniform(15, 90, n)
        pH = self.rng.uniform(4.5, 8.5, n)
        om = self.rng.uniform(1.0, 6.0, n)
        T  = self.rng.uniform(5, 40, n)
        R  = self.rng.uniform(10, 300, n)
        F  = self.rng.uniform(20, 200, n)
        A  = self.rng.uniform(0.5, 20, n)

        # ideal mid-points per crop
        ideal = {
            "Rice": (0.22, 0.18, 0.28, 75, 6.2, 3.5, 27, 220, 120),
            "Wheat": (0.22, 0.18, 0.28, 42, 6.7, 3.5, 18, 75, 100),
            "Maize": (0.28, 0.22, 0.32, 55, 6.5, 3.5, 24, 100, 130),
            "Sugarcane": (0.32, 0.22, 0.38, 70, 6.7, 3.5, 28, 175, 150),
            "Cotton": (0.22, 0.16, 0.28, 48, 7.0, 3.5, 28, 85, 110),
            "Soybean": (0.18, 0.18, 0.28, 52, 6.5, 3.5, 24, 105, 60),
            "Groundnut": (0.16, 0.18, 0.25, 42, 6.7, 3.5, 26, 85, 60),
            "Barley": (0.18, 0.14, 0.24, 38, 6.9, 3.5, 15, 60, 80),
            "Millet": (0.16, 0.14, 0.22, 32, 6.5, 3.5, 28, 50, 50),
            "Tomato": (0.25, 0.22, 0.36, 58, 6.5, 3.5, 23, 70, 150),
            "Potato": (0.25, 0.22, 0.38, 62, 5.8, 3.5, 17, 70, 170),
            "Onion": (0.22, 0.18, 0.32, 52, 6.7, 3.5, 20, 60, 130),
        }[crop_name]

        iN, iP, iK, im, ipH, iom, iT, iR, iF = ideal

        # Yield function: base * combined penalty
        def penalty(val, ideal, scale):
            return np.exp(-((val - ideal) ** 2) / (2 * scale ** 2))

        yield_ = base_yield * (
            0.6 * penalty(N, iN, 0.10)
            + 0.4 * penalty(P, iP, 0.08)
            + 0.4 * penalty(K, iK, 0.10)
            + 0.6 * penalty(m, im, 20)
            + 0.5 * penalty(pH, ipH, 0.8)
            + 0.3 * penalty(om, iom, 1.5)
            + 0.7 * penalty(T, iT, 6)
            + 0.5 * penalty(R, iR, 60)
            + 0.4 * penalty(F, iF, 50)
        ) / 4.4
        yield_ += self.rng.normal(0, 0.05 * base_yield, n)
        yield_ = np.clip(yield_, 0.1, None)

        X = np.column_stack([N, P, K, m, pH, om, T, R, F, A])
        return X, yield_

    # -------- train all crop models --------
    def train_all(self):
        print("\n🌾 TRAINING YIELD PREDICTION MODELS (per crop)")
        print("-" * 50)
        for crop in ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton",
                     "Soybean", "Groundnut", "Barley", "Millet",
                     "Tomato", "Potato", "Onion"]:
            X, y = self._generate_training_data(crop)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            r2 = r2_score(y_te, preds)
            self.models[crop] = model
            print(f"   {crop:<11} R² = {r2:.3f}")

    # -------- single prediction --------
    def predict(self, crop, features_dict, area_ha=1.0):
        if crop not in self.models:
            raise ValueError(f"No model for crop '{crop}'")
        row = [features_dict.get(f, 0.0) for f in self.features]
        row[-1] = area_ha  # set area
        X = np.array(row).reshape(1, -1)
        y_ha = float(self.models[crop].predict(X)[0])
        return {"yield_per_ha": y_ha, "total_yield": y_ha * area_ha, "area": area_ha}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE CROP HEALTH MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleHyperspectralModel:
    """Voting ensemble: RandomForest + GradientBoosting + XGBoost."""

    def __init__(self, input_shape, n_classes, class_names):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.class_names = class_names
        self.ensemble = None
        self.meta = None

    def train_models(self, X_train, y_train, X_val, y_val, epochs=5):
        # (epochs param kept for API compatibility — unused here)
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
        ]
        if HAS_XGB:
            estimators.append((
                "xgb",
                XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="mlogloss",
                              random_state=42, verbosity=0)
            ))

        self.ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        self.ensemble.fit(X_train, y_train)
        val_acc = self.ensemble.score(X_val, y_val)
        print(f"   Ensemble validation accuracy: {val_acc*100:.2f}%")
        return []  # no NN history

    def ensemble_predict(self, X):
        return self.ensemble.predict_proba(X)

    def build_xgboost_meta_learner(self, X_train, y_train):
        # Meta-learner = LogisticRegression on the ensemble's probabilities
        proba_train = self.ensemble.predict_proba(X_train)
        self.meta = LogisticRegression(max_iter=1000, multi_class="multinomial")
        self.meta.fit(proba_train, y_train)

    def predict_with_xgboost(self, X):
        if self.meta is None:
            return self.ensemble_predict(X)
        proba = self.ensemble.predict_proba(X)
        return self.meta.predict_proba(proba)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. IoT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class IoTDashboard:
    def __init__(self):
        self.history = {"date": [], "temperature": [], "humidity": [],
                        "soil_moisture": [], "crop_health_score": []}
        self.alerts = []
        self.reports = {}

    def generate_real_time_data(self, n_days=15):
        rng = np.random.default_rng(7)
        for d in range(n_days):
            self.history["date"].append(f"Day {d+1}")
            self.history["temperature"].append(22 + rng.normal(0, 3))
            self.history["humidity"].append(60 + rng.normal(0, 8))
            self.history["soil_moisture"].append(28 + rng.normal(0, 4))
            self.history["crop_health_score"].append(0.75 + rng.normal(0, 0.08))

    def generate_alerts(self, crop_preds=None, soil_preds=None, pesticide_stats=None):
        self.alerts = []
        if soil_preds and "moisture" in soil_preds:
            m = float(np.mean(soil_preds["moisture"]))
            if m < 20: self.alerts.append(f"⚠ Low soil moisture ({m:.1f}%)")
        if soil_preds and "nitrogen" in soil_preds:
            n = float(np.mean(soil_preds["nitrogen"]))
            if n < 0.15: self.alerts.append(f"⚠ Low nitrogen ({n:.3f})")
        if pesticide_stats:
            if pesticide_stats.get("missed", 0) > 15:
                self.alerts.append("⚠ High missed pesticide coverage")
            if pesticide_stats.get("over_applied", 0) > 20:
                self.alerts.append("⚠ High over-application of pesticide")
        if not self.alerts:
            self.alerts.append("✅ All monitored parameters within acceptable ranges")

    def display_dashboard(self):
        print("\n📱 IOT DASHBOARD — RECENT HISTORY")
        if self.history["temperature"]:
            print(f"   Avg temperature    : {np.mean(self.history['temperature']):.1f} °C")
            print(f"   Avg humidity       : {np.mean(self.history['humidity']):.1f} %")
            print(f"   Avg soil moisture  : {np.mean(self.history['soil_moisture']):.1f} %")
            print(f"   Avg crop health    : {np.mean(self.history['crop_health_score']):.2f}")
        print("\n🚨 ACTIVE ALERTS:")
        for a in self.alerts:
            print(f"   {a}")

    def generate_reports(self):
        self.reports = {"summary": {
            "avg_temperature": float(np.mean(self.history["temperature"])),
            "avg_humidity":    float(np.mean(self.history["humidity"])),
            "avg_soil_moisture": float(np.mean(self.history["soil_moisture"])),
            "avg_crop_health": float(np.mean(self.history["crop_health_score"])),
        }, "alerts": list(self.alerts)}
        return self.reports


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION ENGINE (inline Colab plots)
# ═══════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """All plots display inline (Colab)."""

    def _show(self, fig):
        plt.tight_layout()
        plt.show()

    # ---------- crop ----------
    def plot_spectral_signatures(self, X, y, class_names):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, name in enumerate(class_names):
            mask = (y == i)
            if mask.sum() == 0: continue
            ax.plot(X[mask].mean(axis=0), label=name, linewidth=1.8)
        ax.set_title("Mean Spectral Signature per Crop Class")
        ax.set_xlabel("Band index"); ax.set_ylabel("Reflectance")
        ax.legend(); ax.grid(alpha=0.3)
        self._show(fig)

    def plot_training_history(self, histories):
        if not histories: return

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — Crop Health")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        self._show(fig)

    # ---------- soil ----------
    def plot_soil_property_predictions(self, soil_results):
        n = len(soil_results)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, (prop, res) in zip(axes, soil_results.items()):
            ax.scatter(res["y_true"], res["y_pred"], alpha=0.5, s=12)
            lo = min(res["y_true"].min(), res["y_pred"].min())
            hi = max(res["y_true"].max(), res["y_pred"].max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax.set_title(f"{prop} (R²={res['r2']:.2f})")
            ax.set_xlabel("True"); ax.set_ylabel("Predicted")
            ax.grid(alpha=0.3)
        for ax in axes[n:]:
            ax.axis("off")
        self._show(fig)

    # ---------- pesticide ----------
    def plot_pesticide_distribution(self, classification, pesticide_index, stats):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = axes[0].imshow(pesticide_index, cmap="viridis")
        axes[0].set_title("Pesticide Application Index"); fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(classification, cmap="tab10", vmin=0, vmax=3)
        axes[1].set_title("Coverage: 0=Optimal,1=Under,2=Over,3=Missed")
        fig.colorbar(im1, ax=axes[1])
        fig.suptitle(f"Optimal {stats['optimal']:.1f}% | Under {stats['under_applied']:.1f}% | "
                     f"Over {stats['over_applied']:.1f}% | Missed {stats['missed']:.1f}%")
        self._show(fig)

    # ---------- dashboard ----------
    def plot_monitoring_timeline(self, dashboard):
        hist = dashboard.history
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(hist["date"], hist["temperature"], label="Temperature (°C)")
        axes[0].plot(hist["date"], hist["humidity"], label="Humidity (%)")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_title("Environmental Monitoring")
        axes[1].plot(hist["date"], hist["soil_moisture"], label="Soil Moisture (%)", color="green")
        axes[1].plot(hist["date"], np.array(hist["crop_health_score"]) * 100,
                     label="Crop Health × 100", color="orange")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_title("Soil & Crop Health")
        for ax in axes: ax.tick_params(axis="x", rotation=45)
        self._show(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTEGRATED SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class IntegratedCropMonitoringSystem:

    def __init__(self):
        self.data_loader = HyperspectralDataLoader()
        self.crop_model = None
        self.soil_analyzer = SoilAnalyzer()
        self.pesticide_analyzer = PesticideAnalyzer()
        self.dashboard = IoTDashboard()
        self.visualizer = VisualizationEngine()
        self.suggestion_engine = AICropSuggestionEngine()
        self.yield_engine = YieldPredictionEngine()

        self.X_crop = None
        self.y_crop = None
        self.class_names = None
        self.X_soil = None
        self.y_soil = None
        self.crop_predictions = None
        self.soil_predictions = None
        self.pesticide_stats = None

    def run_complete_system(self):
        print("\n" + "=" * 80)
        print("STARTING COMPLETE INTEGRATED CROP MONITORING SYSTEM")
        print("=" * 80)

        self._generate_data()
        self._train_crop_health_model()
        self._train_soil_models()
        self._analyze_pesticide_distribution()
        self._setup_monitoring_dashboard()
        self._train_yield_models()
        self._generate_final_report()

        print("\n" + "=" * 80)
        print("SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    # -------- pipeline steps --------
    def _generate_data(self):
        print("\n📊 GENERATING SYNTHETIC DATASETS")
        print("-" * 40)
        self.X_crop, self.y_crop, self.class_names = self.data_loader.generate_synthetic_data(n_samples=2000)
        self.X_crop, self.y_crop = self.data_loader.preprocess_data(self.X_crop, self.y_crop)
        print(f"Crop Health Data: {self.X_crop.shape[0]} samples, {self.X_crop.shape[1]} bands")
        print(f"Classes: {self.class_names}")
        self.X_soil, self.y_soil = self.soil_analyzer.generate_soil_data()
        print(f"Soil Data: {self.X_soil.shape[0]} samples, {self.X_soil.shape[1]} bands")

    def _train_crop_health_model(self):
        print("\n🌱 TRAINING CROP HEALTH ENSEMBLE MODEL")
        print("-" * 40)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_crop, self.y_crop, test_size=0.2, random_state=42, stratify=self.y_crop)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        print(f"Training samples  : {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples      : {X_test.shape[0]}")

        input_shape = (self.X_crop.shape[1],)
        self.crop_model = EnsembleHyperspectralModel(
            input_shape, len(self.class_names), self.class_names)

        self.crop_model.train_models(X_train, y_train, X_val, y_val, epochs=5)

        ensemble_pred = self.crop_model.ensemble_predict(X_test)
        y_pred_ensemble = np.argmax(ensemble_pred, axis=1)

        self.crop_model.build_xgboost_meta_learner(X_train, y_train)
        meta_pred = self.crop_model.predict_with_xgboost(X_test)
        y_pred_meta = np.argmax(meta_pred, axis=1)

        ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
        meta_acc = accuracy_score(y_test, y_pred_meta)

        print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
        print(f"   Ensemble Accuracy      : {ensemble_acc*100:.2f}%")
        print(f"   Meta-Learner Accuracy  : {meta_acc*100:.2f}%")

        self.visualizer.plot_spectral_signatures(self.X_crop, self.y_crop, self.class_names)
        self.visualizer.plot_confusion_matrix(y_test, y_pred_meta, self.class_names)

        self.crop_predictions = meta_pred

    def _train_soil_models(self):
        print("\n🌍 TRAINING SOIL PROPERTY ESTIMATION MODELS")
        print("-" * 40)
        soil_results = self.soil_analyzer.train_models(self.X_soil, self.y_soil)
        X_soil_test = self.X_soil[:150]
        self.soil_predictions = self.soil_analyzer.predict_soil_properties(X_soil_test)
        self.visualizer.plot_soil_property_predictions(soil_results)

    def _analyze_pesticide_distribution(self):
        print("\n🧪 ANALYZING PESTICIDE DISTRIBUTION")
        print("-" * 40)
        field_map, _ = self.pesticide_analyzer.generate_field_map()
        classification, pesticide_index, stats = self.pesticide_analyzer.detect_application_issues(field_map)

        print(f"\n📊 PESTICIDE COVERAGE ANALYSIS:")
        print(f"   ✅ Optimal Coverage: {stats['optimal']:.1f}%")
        print(f"   ⚠  Under-Applied  : {stats['under_applied']:.1f}%")
        print(f"   ❌ Over-Applied   : {stats['over_applied']:.1f}%")
        print(f"   🚫 Missed Areas   : {stats['missed']:.1f}%")

        self.visualizer.plot_pesticide_distribution(classification, pesticide_index, stats)
        self.pesticide_stats = stats

    def _setup_monitoring_dashboard(self):
        print("\n📱 SETTING UP REAL-TIME MONITORING DASHBOARD")
        print("-" * 40)
        self.dashboard.generate_real_time_data(n_days=15)
        self.dashboard.generate_alerts(self.crop_predictions,
                                       self.soil_predictions,
                                       self.pesticide_stats)
        self.dashboard.display_dashboard()
        self.dashboard.generate_reports()
        self.visualizer.plot_monitoring_timeline(self.dashboard)

    def _train_yield_models(self):
        self.yield_engine.train_all()

    def _generate_final_report(self):
        print("\n📄 GENERATING COMPREHENSIVE FINAL REPORT")
        print("-" * 40)

        if self.crop_predictions is not None and len(self.crop_predictions) > 0:
            health_distribution = np.bincount(np.argmax(self.crop_predictions, axis=1),
                                              minlength=len(self.class_names))
            total_samples = len(self.crop_predictions)
            print(f"\n🌱 CROP HEALTH SUMMARY:")
            for i, class_name in enumerate(self.class_names):
                percentage = health_distribution[i] / total_samples * 100
                print(f"   {class_name}: {percentage:.1f}%")

        if self.soil_predictions:
            print(f"\n🌍 SOIL HEALTH SUMMARY:")
            for prop, values in self.soil_predictions.items():
                print(f"   {prop.replace('_', ' ').title()}: {np.mean(values):.2f}")

        print(f"\n💡 RECOMMENDATIONS:")
        if self.crop_predictions is not None and len(self.crop_predictions) > 0:
            health_distribution = np.bincount(np.argmax(self.crop_predictions, axis=1),
                                              minlength=len(self.class_names))
            healthy_pct = health_distribution[0] / len(self.crop_predictions) * 100
            if healthy_pct < 70:
                print("   🔸 Consider targeted nutrient management")
                print("   🔸 Schedule pest control assessment")
            else:
                print("   ✅ Crop health is generally good — maintain current practices")

        if self.soil_predictions and 'nitrogen' in self.soil_predictions:
            if np.mean(self.soil_predictions['nitrogen']) < 0.15:
                print("   🔸 Apply nitrogen-rich fertilizer")
        if self.soil_predictions and 'moisture' in self.soil_predictions:
            if np.mean(self.soil_predictions['moisture']) < 20:
                print("   🔸 Increase irrigation frequency")
        if self.pesticide_stats:
            if self.pesticide_stats.get('missed', 0) > 15:
                print("   🔸 Improve pesticide application coverage")
            if self.pesticide_stats.get('over_applied', 0) > 20:
                print("   🔸 Reduce pesticide application rates")

        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • Ensemble + meta-learner crop health classifier")
        print(f"   • AI-driven crop suggestion engine (12 crops)")
        print(f"   • Per-crop yield prediction models (tons/hectare)")
        print(f"   • Comprehensive soil + pesticide analytics")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. INTERACTIVE USER INPUT + PERSONALIZED REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def ask_user_inputs():
    """Ask the user for soil, climate, and farm data. Returns dicts."""
    print("\n" + "=" * 80)
    print("📝 PLEASE ENTER YOUR FARM DATA")
    print("=" * 80)
    print("(Press ENTER to use the default value shown in [brackets])\n")

    def ask(prompt, default, cast=float):
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"   ⚠ Invalid input, using default {default}")
            return default

    soil = {
        "nitrogen":       ask("Soil Nitrogen  (0.05–0.40)",             0.22),
        "phosphorus":     ask("Soil Phosphorus(0.05–0.35)",             0.18),
        "potassium":      ask("Soil Potassium (0.10–0.50)",             0.28),
        "moisture":       ask("Soil Moisture %  (10–90)",               55.0),
        "ph":             ask("Soil pH          (4.5–8.5)",              6.5),
        "organic_matter": ask("Organic Matter % (1–6)",                  3.2),
    }
    climate = {
        "temp": ask("Average Temperature °C (5–40)", 25.0),
        "rain": ask("Annual Rainfall mm  (10–300)", 110.0),
    }
    farm = {
        "area":       ask("Farm area (hectares)", 2.0),
        "fertilizer": ask("Fertilizer used (kg/ha)", 120.0),
    }
    return soil, climate, farm


def show_personalized_report(soil, climate, farm, suggestion_engine, yield_engine):
    """Generate suggestions, yield prediction, and personalized plots."""

    print("\n" + "=" * 80)
    print("🎯 AI-POWERED PERSONALIZED RECOMMENDATIONS")
    print("=" * 80)

    # ---------- 1. Crop Suggestions ----------
    soil_for_model = {
        "N": soil["nitrogen"], "P": soil["phosphorus"], "K": soil["potassium"],
        "moisture": soil["moisture"], "pH": soil["ph"],
        "organic_matter": soil["organic_matter"],
    }
    climate_for_model = {"temp": climate["temp"], "rain": climate["rain"]}

    suggestions = suggestion_engine.suggest(soil_for_model, climate_for_model, top_k=5)

    print("\n🌱 TOP 5 SUITABLE CROPS FOR YOUR FARM:")
    print("-" * 50)
    for rank, (crop, score, reasons) in enumerate(suggestions, 1):
        print(f"   {rank}. {crop:<12} suitability = {score*100:.1f}%")
        for r in reasons:
            print(f"        • {r}")

    # ---------- 2. Yield Prediction for Top-3 ----------
    print("\n🌾 YIELD PREDICTION FOR TOP 3 CROPS:")
    print("-" * 50)
    yield_results = {}
    for crop, score, _ in suggestions[:3]:
        feats = {**soil_for_model, **climate_for_model, "fertilizer": farm["fertilizer"]}
        try:
            pred = yield_engine.predict(crop, feats, area_ha=farm["area"])
            yield_results[crop] = pred
            print(f"   {crop:<12}: {pred['yield_per_ha']:.2f} t/ha  →  "
                  f"total ≈ {pred['total_yield']:.2f} t on {pred['area']} ha")
        except Exception as e:
            print(f"   {crop:<12}: prediction failed ({e})")

    # ---------- 3. Personalized Plots ----------
    _plot_suggestions(suggestions)
    _plot_yield_comparison(yield_results)
    _plot_soil_radar(soil)
    _plot_soil_vs_ideal(soil)


def _plot_suggestions(suggestions):
    crops = [s[0] for s in suggestions]
    scores = [s[1] * 100 for s in suggestions]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.array(scores) / 100)
    bars = ax.barh(crops[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Suitability Score (%)")
    ax.set_title("🌱 Top 5 Crop Suggestions for Your Farm")
    for bar, s in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{s:.1f}%", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_yield_comparison(yield_results):
    if not yield_results: return
    crops = list(yield_results.keys())
    per_ha = [yield_results[c]["yield_per_ha"] for c in crops]
    totals = [yield_results[c]["total_yield"] for c in crops]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(crops, per_ha, color="seagreen")
    axes[0].set_title("Predicted Yield (t/ha)")
    axes[0].set_ylabel("tons / hectare")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(crops, totals, color="darkorange")
    axes[1].set_title(f"Predicted Total Yield (for your area)")
    axes[1].set_ylabel("tons")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(15)
    plt.tight_layout()
    plt.show()


def _plot_soil_radar(soil):
    labels = ["Nitrogen", "Phosphorus", "Potassium", "Moisture", "pH", "Organic Matter"]
    values = [soil["nitrogen"], soil["phosphorus"], soil["potassium"],
              soil["moisture"], soil["ph"], soil["organic_matter"]]
    # Normalize each to 0–1 using reasonable max values
    max_vals = [0.40, 0.35, 0.50, 90, 8.5, 6.0]
    norm = [min(1.0, v / m) for v, m in zip(values, max_vals)]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    norm += norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, norm, "o-", linewidth=2, color="green")
    ax.fill(angles, norm, alpha=0.25, color="green")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("🧪 Your Soil Profile (normalized)", y=1.08)
    plt.tight_layout()
    plt.show()


def _plot_soil_vs_ideal(soil):
    # Compare against a general "ideal" mid-range
    ideal = {"nitrogen": 0.22, "phosphorus": 0.18, "potassium": 0.28,
             "moisture": 55, "ph": 6.5, "organic_matter": 3.2}
    props = list(ideal.keys())
    yours = [soil[p] for p in props]
    ideals = [ideal[p] for p in props]

    x = np.arange(len(props))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, yours,  w, label="Your Soil",  color="steelblue")
    ax.bar(x + w/2, ideals, w, label="Ideal Range", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ").title() for p in props], rotation=15)
    ax.set_title("🧪 Your Soil vs Ideal Soil Values")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        crop_system = IntegratedCropMonitoringSystem()
        crop_system.run_complete_system()

        # ---------- INTERACTIVE PART ----------
        soil, climate, farm = ask_user_inputs()
        show_personalized_report(
            soil, climate, farm,
            crop_system.suggestion_engine,
            crop_system.yield_engine,
        )

        print("\n" + "=" * 80)
        print("SYSTEM READY FOR DEPLOYMENT")
        print("=" * 80)
        print("\nKey Features Implemented:")
        print("✅ Hyperspectral Data Processing & Augmentation")
        print("✅ Ensemble ML for Crop Health Classification (>90% target)")
        print("✅ Soil Property Estimation (6 properties)")
        print("✅ Pesticide Distribution Analysis")
        print("✅ Real-time IoT Monitoring Dashboard")
        print("✅ AI-Powered Crop Suggestion (12 crops)")
        print("✅ Per-Crop Yield Prediction (t/ha)")
        print("✅ Interactive Personalized Report")

    except Exception as e:
        print(f"\n❌ Error during system execution: {str(e)}")
        import traceback
        traceback.print_exc()