from __future__ import annotations

import shap
import numpy as np
import pandas as pd
from typing import Any


class ShapExplainer:
    def __init__(self, model: Any):
        self.model = model
        self.explainer = None

    def fit(self, X_background: pd.DataFrame):
        if hasattr(self.model, "named_steps") and "clf" in self.model.named_steps:
            model = self.model.named_steps["clf"]
        else:
            model = self.model

        if model.__class__.__name__.startswith("XGB"):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model.predict_proba, X_background)
        return self

    def shap_values(self, X: pd.DataFrame):
        if self.explainer is None:
            raise RuntimeError("Call fit() first with background data")
        return self.explainer(X)
