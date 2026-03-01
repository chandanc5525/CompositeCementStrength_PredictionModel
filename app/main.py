from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import ConcreteInput
from app.service import PredictionService


app = FastAPI(
    title="Cement Strength Prediction API",
    version="1.0.0"
)

# Static + Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model once
service = PredictionService()



# Health Check

@app.get("/health")
def health_check():
    return {"status": "API is running"}



# UI Home Page

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "main.html",
        {"request": request}
    )



# Form Prediction

@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    cement: float = Form(...),
    blast_furnace_slag: float = Form(...),
    fly_ash: float = Form(...),
    water: float = Form(...),
    superplasticizer: float = Form(...),
    coarse_aggregate: float = Form(...),
    fine_aggregate: float = Form(...),
    age: int = Form(...)
):

    input_data = ConcreteInput(
        cement=cement,
        blast_furnace_slag=blast_furnace_slag,
        fly_ash=fly_ash,
        water=water,
        superplasticizer=superplasticizer,
        coarse_aggregate=coarse_aggregate,
        fine_aggregate=fine_aggregate,
        age=age
    )

    prediction = service.predict(input_data)

    return templates.TemplateResponse(
        "main.html",
        {
            "request": request,
            "prediction": round(prediction, 2)
        }
    )



# JSON API Prediction

@app.post("/predict_json")
def predict_json(input_data: ConcreteInput):

    prediction = service.predict(input_data)

    return JSONResponse(
        content={
            "predicted_strength_mpa": round(prediction, 2)
        }
    )