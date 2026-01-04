# Schilderijen Classificeren met Deep Learning

Dit project focust op het automatisch classificeren van schilderijen op basis van de schilder met behulp van **deep learning**. Door gebruik te maken van **convolutionele neurale netwerken** en **transfer learning met VGG16** wordt onderzocht in welke mate schilderijen van verschillende kunstenaars automatisch onderscheiden kunnen worden.

Dit project werd uitgevoerd als onderdeel van een **Deep Learning-opdracht** binnen de opleiding **Bachelor Toegepaste Informatica – specialisatie Artificiële Intelligentie**.

---

## Doel van het project

Het doel van dit project is het ontwikkelen van een model dat schilderijen correct kan classificeren per schilder op basis van visuele kenmerken zoals kleurgebruik, textuur en compositie. Daarbij werden volgende aspecten onderzocht:

* verzamelen en opschonen van beelddata
* omgaan met klasse-onbalans
* vergelijken van eenvoudige ConvNet-modellen en transfer learning
* toepassen van data augmentation en fine-tuning
* evaluatie met geschikte metriek (accuracy, F1-score, confusion matrix)
* praktische toepasbaarheid via een demo-applicatie

---

## Projectstructuur

```text
opdracht schilderijen classificeren/
│
├── schilderijen/
│   ├── Picasso/
│   ├── Rubens/
│   ├── Mondriaan/
│   └── Rembrandt/
│   └── (opgeschoonde afbeeldingen per schilder)
│
├── datasets/
│   └── alle_schilders/
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   ├── scraping.ipynb
│   └── alle schilders/
│       ├── alle_schilders.ipynb
│       ├── demo_gradio.ipynb
│       │
│       ├── simple no augmentation/
│       ├── simple with augmentation/
│       ├── vgg16_feature_extraction/
│       ├── vgg16_feature_extraction_aug/
│       ├── vgg16_finetuning/
│       └── vgg16_finetuning_class_weights/
│
├── venv_tf215/
│   └── Python 3.11 virtual environment
│
├── .gitignore
└── README.md
```

---

## Gebruikte omgevingen

### Lokale omgeving

Lokaal werd gewerkt voor:

* data scraping en preprocessing
* notebookontwikkeling
* analyse van resultaten
* ontwikkeling en testen van de Gradio demo

**Belangrijk:**
Het project vereist **Python 3.11** en **TensorFlow 2.15** voor compatibiliteit met het finale model.

### HPC-omgeving (Vlaamse Supercomputer)

De training van deep learning-modellen gebeurde op de Vlaamse Supercomputer, aangezien GPU-ondersteuning noodzakelijk was.
Per experiment werd een apart Python-script uitgevoerd via **SLURM jobs**. Resultaten (metrics, plots en confusion matrices) werden systematisch per experiment opgeslagen.

---

## Installatie (lokaal)

```bash
python3.11 -m venv venv_tf215
venv_tf215\Scripts\activate
pip install tensorflow==2.15 gradio matplotlib scikit-learn numpy
```

> TensorFlow 2.15 is **niet compatibel met Python 3.12**

---

## Data verzamelen en voorbereiden

* De schilderijen van **Picasso, Rubens en Mondriaan** werden aangeleverd.
* De schilderijen van **Rembrandt** werden zelf verzameld via web scraping.
* Het notebook `scraping.ipynb` werd gebruikt om afbeeldingen van Rembrandt automatisch te downloaden.
* Alle afbeeldingen werden opgeschoond en opgeslagen per schilder in de map `schilderijen/`.
* Op basis van deze cleaned data werd de dataset opgesplitst in train-, validatie- en testsets in `datasets/alle_schilders/`.

---

## Modellen & experimenten

De volgende modellen en stappen werden onderzocht:

1. Eenvoudig ConvNet zonder data augmentation
2. Eenvoudig ConvNet met data augmentation
3. VGG16 feature extraction
4. VGG16 feature extraction met data augmentation
5. VGG16 fine-tuning (block5)
6. VGG16 fine-tuning met class weights *(finaal model)*

Elke experimentmap bevat:

* het gebruikte Python-script
* het beste getrainde model
* evaluatiemetrics en plots
* confusion matrix en classification report

---

## Resultaten (finaal model)

* Test accuracy: **±95.7%**
* Macro F1-score: **±0.935**
* Evenwichtige prestaties over alle klassen
* Verbeterde herkenning van minderheidsklassen door class weighting

---

## Gradio demo

Een Gradio-applicatie werd ontwikkeld om het finale model te demonstreren.
De gebruiker kan een schilderij uploaden en krijgt de voorspelde schilder en de bijhorende waarschijnlijkheden te zien.

Om compatibiliteitsproblemen te vermijden werd het finale model lokaal opgeslagen in **`.h5`-formaat**, hoewel het oorspronkelijk op de HPC-omgeving in `.keras`-formaat werd getraind.

---

## Auteur

Naam: **Moustafa**
Opleiding: Bachelor Toegepaste Informatica – Artificiële Intelligentie
Instelling: **VIVES**

---

## Context

Dit project werd uitgevoerd als onderdeel van een **Deep Learning-opdracht** en is bedoeld voor educatieve doeleinden.

