import modal

image = modal.Image.debian_slim().pip_install("fastapi","diffusers","transformers","accelerate","peft")
app = modal.App("Pentagram", image=image)

with image.imports():
    from diffusers import AutoPipelineForText2Image
    import torch
    import io
    from fastapi import Response

@app.cls(image=image,gpu="A10G")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

    # @modal.build()
    # @modal.enter()
    # def load_lora_weights(self):
    #     self.pipe.load_lora_weights("nerijs/pixel-art-xl", adapter_name="pixel")
    #     self.pipe.set_adapters(["pixel"], adapter_weights=[1.2])

    @modal.web_endpoint()
    def generate(self, prompt = "pixel 20 . A cinematic shot of a baby racoon wearing an intricate italian priest robe."):
        image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        buffer = io.BytesIO()
        image.save(buffer,format="JPEG")
        return Response(content=buffer.getvalue(),media_type='image/jpeg')






# @app.function()
# @modal.web_endpoint(label="generateImage")
# def mySimpleEndPoint():
#     return "Hello world!"

# app = modal.App("example-get-started")

# @app.function()
# def square(x):
#     print("This code is running on a remote worker!")
#     return x**2

# @app.local_entrypoint()
# def main():
#     print("the square is", square.remote(42))
