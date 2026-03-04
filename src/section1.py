import os
from PIL import Image

import numpy as np
import torchvision
import pickle

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models
import matplotlib.pyplot as plt

def s1(nameim):
    # 1. Garder l'image originale pour l'affichage final
    img_pil = Image.open(nameim).convert('RGB')

    # Chargement des classes
    imagenet_classes = pickle.load(open('../data/imagenet_classes.pkl', 'rb'))

    # 2. Préparation pour le réseau (Calcul)
    img_tensor = img_pil.resize((224, 224), Image.BILINEAR)
    img_tensor = np.array(img_tensor, dtype=np.float32) / 255
    img_tensor = img_tensor.transpose((2, 0, 1))

    mu = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    sigma = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    x = (torch.Tensor(img_tensor) - mu) / sigma
    x = x.unsqueeze(0)

    # 3. Prédiction
    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()

    with torch.no_grad():
        y = vgg16(x)

    pred = np.argmax(y.numpy())
    class_name = imagenet_classes[pred]
    print('Predicted class: %d - %s' % (pred, class_name))

    # 4. Sauvegarde
    output_dir = '../plot'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. Affichage et Sauvegarde
    plt.figure(figsize=(8, 6))
    plt.imshow(img_pil)
    plt.title(f"Prediction: {class_name} (Class {pred})")
    plt.axis('off')

    base_name = os.path.basename(nameim).split('.')[0]
    save_path = os.path.join(output_dir, f'prediction_{base_name}.png')

    plt.savefig(save_path)
    print(f"Plot de prédiction sauvegardé sous : {save_path}")
    plt.show()


def s1_4(nameim):
    # 1. Chargement et Prétraitement (identique à s1 pour la cohérence)
    img = Image.open(nameim).convert('RGB')
    img_resized = img.resize((224, 224), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255
    img_np = img_np.transpose((2, 0, 1))

    mu = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    sigma = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (torch.Tensor(img_np) - mu) / sigma
    img_tensor = img_tensor.unsqueeze(0)

    # 2. Chargement du modèle VGG16
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()

    # 3. Extraction de la première couche convolutionnelle
    # Dans VGG16, 'features' est un séquentiel. L'index 0 est la première Conv2d.
    first_conv_layer = vgg16.features[0]

    with torch.no_grad():
        activations = first_conv_layer(img_tensor).squeeze(0).cpu().numpy()

        output_dir = '../plot'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_filters = activations.shape[0]
        plt.figure(figsize=(20, 20))

        for i in range(num_filters):
            plt.subplot(8, 8, i + 1)
            plt.imshow(activations[i], cmap='viridis')
            plt.title(f'F#{i}', fontsize=8)
            plt.axis('off')

        plt.suptitle(f"Activations - Couche 1 - {nameim}", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Définition du nom de fichier (basé sur le nom de l'image d'entrée)
        base_name = os.path.basename(nameim).split('.')[0]
        save_path = os.path.join(output_dir, f'activations_{base_name}.png')

        # Sauvegarde du plot
        plt.savefig(save_path)
        print(f"Image sauvegardée sous : {save_path}")

        plt.show()