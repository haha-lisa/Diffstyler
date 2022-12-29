import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./LPIPS/biggan+CLIP/biggan_ex_dir0/4.png")).unsqueeze(0).to(device)
text = clip.tokenize(["A stunning natural landscape painting is created by an artist Paul Cezanne in post-impressionism style.","This is a gorgeous natural landscape painting created by the well-known artist Henri Matisse in pointillism and neo-impressionism styles.", "This is a colorful painting of mountains in the style of futurism drawn by Natalia Goncharova.","This is an oil painting of flowers in the style of impressionism drawn by Pierre-Auguste Renoir.","The beautiful painting depicts the interior of a golden and regal palace drawn by artist Karlsimon.","Claude Monet's painting of trees in grey weather in Impressionism style.","A painting of cityscape drawn by Bernardo Bellotto in rococo style.","The painting depicts a magnificent giant castle sitting on a lake in the style of fauvism and post impressionism drawn by Henri Matisse."]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(similarity)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("./LPIPS/biggan+CLIP/biggan_ex_dir0/2.png")).unsqueeze(0).to(device)
# image2 = preprocess(Image.open("./LPIPS/fusedream/fusedream_ex_dir1/2.png")).unsqueeze(0).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     image2_features = model.encode_image(image2)
  
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     image2_features /= image2_features.norm(dim=-1, keepdim=True)
#     similarity = (100.0 * image_features @ image2_features).softmax(dim=-1).sum()
#     print(similarity)
