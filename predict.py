import torch
import torch.nn.functional as F
from torchvision import transforms, models
import argparse
import json
from PIL import Image
from utils import (
    load_categories,
    device_type)

def process_image(image):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    
    img = Image.open(image)
    img_tensor = preprocess(img)
    return img_tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store')
    parser.add_argument('checkpoint', action='store')
    parser.add_argument('--top_k', dest='top_k', type=int, default=1)
    parser.add_argument('--category_names', dest="category_names", default="cat_to_name.json")
    parser.add_argument('--gpu', action="store_true", default=False)
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    if checkpoint["arch"] == "vgg16":
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint["fc"]
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model
    
def predict(image_path, model, device, topk=1):
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    output = model(processed_image.to(device))
    probs, indices = torch.topk(F.softmax(output, dim=1), topk, sorted=True)
    idx_to_class = { v:k for k, v in model.class_to_idx.items()}
    return ([prob.item() for prob in probs[0].data], 
            [idx_to_class[ix.item()] for ix in indices[0].data])

    
def main():
    args = parse_args()
    device = device_type(args.gpu)
    model = load_checkpoint(args.checkpoint)
    model = model.to(device)
    image = args.input
    cat_to_name = load_categories(args.category_names)   
    probs, classes = predict(image, model, device, args.top_k)
    print(probs, [cat_to_name[name] for name in classes])

    
if __name__ == "__main__":
    main()