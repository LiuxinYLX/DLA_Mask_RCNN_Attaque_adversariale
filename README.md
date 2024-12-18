# Mask_RCNN et Attaque adversariale 


## Installation
1. Clonez le dépôt **Mask_RCNN** à partir du site [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
2. Clonez ce dépôt et placez le **attaque_adversariale_bruit.py** dans le répertoire **samples/**.
3. Installez les dépendances :
   ```bash
   pip3 install -r requirements.txt
   ```
4. Exécutez l'installation depuis le répertoire **Mask_RCNN**

    ```bash
    python3 setup.py install
    ``` 
5. Téléchargez les poids pré-entraînés de COCO (mask_rcnn_coco.h5) depuis [releases page](https://github.com/matterport/Mask_RCNN/releases).

## Attaques adversariales
1. Bruit aléatoire
   
   Cette perturbation est implémentée dans le fichier de code **attaque_adversariale_bruit.py**, et tout le processus s'exécute automatiquement avec la commande suivante :
   ```bash
   python3 attaque_adversariale_bruit.py
   ```
2. L'attaque FGSM

   Cette perturbation est implémentée dans le notebook **mrcnn_resnet50_fgsm.ipynb**.
   
Vous trouvez les résultats avant et après l'ajout du bruit aléatoire respectivement dans les répertoires **detection_avant** et **detection_apres_0.01**.
