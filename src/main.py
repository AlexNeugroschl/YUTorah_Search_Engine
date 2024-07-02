from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from .routers.root_router import router

app = FastAPI()

base_router = APIRouter()


@base_router.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TorahNavigator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            p { font-size: 18px; line-height: 1.6; }
        </style>
    </head>
    <body>
        <h1>TorahNavigator</h1>
        <p>
            This project is a recommendation engine for YU Torah, utilizing machine learning
            algorithms to deliver personalized lecture suggestions to users. By analyzing user
            preferences and interaction history, TorahNavigator enhances the learning experience
            by providing tailored content that matches individual interests and study patterns.
        </p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


app.include_router(base_router)
app.include_router(router, prefix="/api/v1", tags=["recommendations"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
