# ProgettoCVeDL
## Nota
Data la dimensione notevole di alcuni file, come gli zip contenenti training e test, e come le variabili dei modelli in cui non si è utilizzato il pooling non è stato possibile caricarli su GitHub. 

Sono però disponibli per essere scaricati nel seguente link: [Scarica i file qui](https://drive.google.com/drive/folders/1XYtHa5A8vpBQtx8DoqDLUFMW6e6g-QQo?usp=sharing)

## Struttura del repository
```bash
    .
    ├──MobileNetV3Base                      #Modello di base da cui siamo partiti
    ├──MobileNetV3_15Epochs                 #Modello con allenamento fatto su sole 15 epoche
    ├──MobileNetV3_NoPooling                #Modello come quello di base ma senza pooling
    ├──MobileNetV3_NoPooling_Tutor          #Come il precedente ma con dataset fornito dal tutor
    ├──MobileNetV3_Rocco_batch2             #Modello con batch-size raddoppiato e dataset del tutor
    ├──MobileNetV3_batch2                   #Modello con batch-size raddoppiato   
    ├──MobileNetV3_lessEpoch                #Modello con allenamento fatto su meno epoche
    ├──MobileNetV3_lrate_1                  #Modello con learning rate pari a 0.001
    ├──MobileNetV3_lrate_2                  #Modello con learning rate pari a 0.0005
    ├──bs64_lr00001_epoch25_EfficentNet     #Modello con rete EfficentNet
    ├──bs64_lr00001_epoch25_ResNet          #Modello con rete ResNet
    ├──bs64_lr00001_epochs30_VGG16          #Modello con rete VGG16
    ├──.gitattributes
    └── README.md
```
