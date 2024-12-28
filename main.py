from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    num_img: int = 2

@app.post("/gen_image")
def gen_image(request: PromptRequest):
    # Extraer el prompt del cuerpo de la solicitud
    prompt = request.prompt
    # Aquí iría la lógica para generar la imagen

    # Cargar el modelo preentrenado de Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    imagenes_base64 = []
    
    for _ in range(request.num_img):
        # Generar la imagen
        image = pipe(prompt).images[0]
        
        # Convertir la imagen a base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        imagenes_base64.append(image_base64)

    # Devolver la lista de imágenes en formato base64
    return {"images": imagenes_base64}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)