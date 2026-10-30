# AI-Based Semiconductor Device Temperature Prediction

A complete capstone starter project for regression-based prediction of semiconductor device temperature.

## Included
- 2,400-row synthetic dataset (`data/semiconductor_temperature_dataset.csv`)
- Feature engineering and training scripts (`src/`)
- Saved best model and metadata (`models/`)
- Test predictions, evaluation metrics, and plots (`outputs/`)
- Streamlit dashboard (`app.py`)
- Project report (`Semiconductor_Temperature_Prediction_Report.docx`)
- Presentation (`Semiconductor_Temperature_Prediction_Presentation.pptx`)

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

## Dataset schema
Target: `device_temperature_C` (°C). Inputs include current, voltage, ambient temperature, load, estimated power loss, thermal resistance, and heat-sink factor. Feature definitions are documented in the report.

## Important limitation
The supplied dataset is synthetic. Its thermal relationship is a simplified illustrative model and is not based on a specific semiconductor datasheet or measured device. For a defensible engineering submission, replace or supplement it with a selected device's datasheet-based LTspice/electrothermal simulation or calibrated low-voltage measurements. Do not use this model for safety-critical decisions.
