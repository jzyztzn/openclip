import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-i18n-256',cache_dir='/home/models/cv/timm/')
tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-i18n-256',cache_dir='/home/models/cv/timm/')

image = Image.open("WechatIMG1008.jpg")
image = preprocess(image).unsqueeze(0)

labels_list = ["a dog", "a cat", "a donut", "a beignet"]
text = tokenizer(labels_list, context_length=model.context_length)
print("image ", image.shape, image[:,:,:3,:3])
print("text ", text.shape, text)
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    print(image_features.shape, image_features[0][:10])
    print(text_features.shape, text_features[0][:10])
    text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)

zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)


## siglip multilingual
print("####" * 100)
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-256-multilingual", cache_dir="/home/models/cv/google/")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual", cache_dir="/home/models/cv/google/")

image = Image.open("WechatIMG1008.jpg")

texts = ["a dog", "a cat", "a donut", "a beignet"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
print(inputs.keys())
print(inputs["pixel_values"].shape, inputs["pixel_values"][:,:,:3,:3])
print(inputs["input_ids"].shape, inputs["input_ids"])
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.keys())
logits_per_text = outputs.text_embeds
logits_per_text = logits_per_text / logits_per_text.norm(dim=-1, keepdim=True)
print(logits_per_text.shape, logits_per_text[0][:10])
logits_per_image_o = outputs.image_embeds
logits_per_image_o = logits_per_image_o / logits_per_image_o.norm(dim=-1, keepdim=True)
print(logits_per_image_o.shape, logits_per_image_o[0][:10])

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")


import numpy as np
# 计算余弦相似度函数
def cosine_similarity(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)  # 点积
    norm1 = np.linalg.norm(vec1)     # 向量 1 的 L2 范数
    norm2 = np.linalg.norm(vec2)     # 向量 2 的 L2 范数
    similarity = dot_product / (norm1 * norm2)
    return similarity

# 计算结果
similarity = cosine_similarity(image_features, logits_per_image_o)
print(similarity)
similarity_text = cosine_similarity(text_features, logits_per_text)
print(similarity_text)