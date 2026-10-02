"""
Zajednički kod za sve tri EuroSAT bilježnice.

Ovdje je sve što trebaju i treniranje, i testiranje, i korištenje modela:
nazivi razreda, priprema slika, model (neuronska mreža) te funkcije za
evaluaciju, spremanje, učitavanje i predviđanje.

U bilježnicama ovu datoteku uvozimo s:  import eurosat_common as ec
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

# --- Osnovne postavke --------------------------------------------------------

# Skup podataka na Hugging Faceu. Na opisu skupa piše "tranganke/eurosat",
# ali to je tipfeler – ispravno ime je "tanganke/eurosat".
DATASET_ID = "tanganke/eurosat"

IMAGE_SIZE = 64                # slike su 64 x 64 piksela
MODEL_PATH = "eurosat_cnn.pt"  # datoteka u koju spremamo naučeni model

# Nazivi razreda – redoslijed odgovara oznakama 0-9 u skupu podataka
CLASS_NAMES = [
    "jednogodišnji usjevi",             # 0 annual crop land
    "šuma",                             # 1 forest
    "grmlje i šikara",                  # 2 brushland or shrubland
    "autocesta ili cesta",              # 3 highway or road
    "industrijske ili poslovne zgrade", # 4 industrial or commercial buildings
    "pašnjak",                          # 5 pasture land
    "trajni nasadi",                    # 6 permanent crop land (voćnjaci, vinogradi)
    "stambene zgrade",                  # 7 residential buildings
    "rijeka",                           # 8 river
    "jezero ili more",                  # 9 lake or sea
]
NUM_CLASSES = len(CLASS_NAMES)

# Iskvarene verzije testnog skupa za test otpornosti: ime dijela skupa -> naziv
CORRUPTIONS = {
    "contrast": "smanjeni kontrast",
    "gaussian_noise": "Gaussov šum",
    "impulse_noise": "impulsni šum",
    "jpeg_compression": "JPEG kompresija",
    "motion_blur": "zamućenje pokretom",
    "pixelate": "pikselizacija",
    "spatter": "mrlje",
}

# --- Priprema slika ----------------------------------------------------------

# Slike za treniranje nasumično mijenjamo (augmentacija), da model vidi više
# različitih primjera i nauči opća pravila umjesto da pamti slike napamet.
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                         # kraća stranica -> 64 piksela
    transforms.CenterCrop(IMAGE_SIZE),                     # kvadrat 64 x 64 iz sredine slike
    transforms.RandomHorizontalFlip(),                     # nasumično zrcali lijevo-desno
    transforms.RandomVerticalFlip(),                       # nasumično zrcali gore-dolje
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # malo promijeni svjetlinu i kontrast
    transforms.ToTensor(),                                 # slika -> tenzor [3, 64, 64], vrijednosti 0 do 1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 0 do 1 -> -1 do 1
])

# Za provjeru i predviđanje sliku samo pripremimo, bez nasumičnih promjena.
# Resize i CenterCrop ne mijenjaju slike iz EuroSAT-a (već su 64 x 64),
# ali omogućuju da model primi i tvoju sliku bilo koje veličine.
eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def tensor_to_image(tensor):
    """Pretvori pripremljeni tenzor [3, 64, 64] natrag u sliku za prikaz (64, 64, 3)."""
    return (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # obrnuto od Normalize


class EuroSATDataset(Dataset):
    """Za zadani redni broj vraća par (pripremljena slika, oznaka razreda)."""

    def __init__(self, hf_split, transform):
        self.data = hf_split        # jedan dio skupa s Hugging Facea, npr. dataset["train"]
        self.transform = transform  # train_transform ili eval_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        image = example["image"].convert("RGB")
        return self.transform(image), example["label"]


# --- Računalo ----------------------------------------------------------------

def get_device():
    """Vrati uređaj za računanje: grafičku karticu (GPU) ako postoji, inače procesor (CPU)."""
    print("PyTorch verzija:", torch.__version__)

    if not torch.cuda.is_available():
        print("UPOZORENJE: grafička kartica (CUDA) nije dostupna – računamo na procesoru (CPU), što je puno sporije.")
        return torch.device("cpu")

    device = torch.device("cuda")
    print("Grafička kartica:", torch.cuda.get_device_name(device))

    # Nove kartice (npr. RTX 5090, oznaka sm_120) trebaju dovoljno novi PyTorch (2.7+ za CUDA 12.8)
    major, minor = torch.cuda.get_device_capability(device)
    supported = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in supported and f"compute_{major}{minor}" not in supported:
        print(f"UPOZORENJE: ovaj PyTorch nije napravljen za ovu karticu (sm_{major}{minor}).")
        print("Instaliraj noviju verziju pa ponovno pokreni kernel (Kernel -> Restart Kernel):")
        print("  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    return device


def mixed_precision(device):
    """Mješovita preciznost: na grafičkoj kartici računaj s brojevima bfloat16 (brže, a dovoljno točno)."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


# --- Model -------------------------------------------------------------------

def conv_block(in_channels, out_channels):
    """Dva konvolucijska sloja koji traže uzorke na slici, pa smanjenje slike na pola."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),  # drži brojeve u ugodnom rasponu -> brže i stabilnije učenje
        nn.ReLU(),                     # negativne vrijednosti postaju 0
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),               # od svaka 2 x 2 piksela zadrži najveću vrijednost
    )


class SimpleCNN(nn.Module):
    """Jednostavna konvolucijska neuronska mreža za 10 razreda EuroSAT-a."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Dio koji "gleda" sliku i izvlači značajke (rubove, teksture, oblike)
        self.features = nn.Sequential(
            conv_block(3, 32),     # 3 boje (RGB) -> 32 mape značajki,  64 x 64 -> 32 x 32
            conv_block(32, 64),    #                                    32 x 32 -> 16 x 16
            conv_block(64, 128),   #                                    16 x 16 ->  8 x 8
            conv_block(128, 256),  #                                     8 x 8  ->  4 x 4
        )
        # Dio koji iz značajki odlučuje kojem razredu slika pripada
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # prosjek svake od 256 mapa značajki -> 256 brojeva
            nn.Flatten(),
            nn.Dropout(0.3),              # tijekom treniranja nasumično "isključi" 30 % brojeva
            nn.Linear(256, num_classes),  # 256 značajki -> 10 rezultata, po jedan za svaki razred
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --- Evaluacija, spremanje, učitavanje, predviđanje --------------------------

@torch.no_grad()  # pri provjeri ne učimo, pa ne trebamo gradijente
def evaluate(model, loader, device):
    """Provjeri model na cijelom skupu. Vraća (prosječni gubitak, točnost, predviđanja, prave oznake)."""
    model.eval()  # način rada za provjeru (isključuje Dropout)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with mixed_precision(device):
            outputs = model(images)
        total_loss += loss_fn(outputs.float(), labels).item()
        all_preds.append(outputs.argmax(dim=1).cpu())  # razred s najvećim rezultatom
        all_labels.append(labels.cpu())

    preds, labels = torch.cat(all_preds), torch.cat(all_labels)
    accuracy = (preds == labels).float().mean().item()
    return total_loss / len(labels), accuracy, preds, labels


def save_model(model, path=MODEL_PATH, **info):
    """Spremi naučene parametre modela i dodatne podatke (npr. epohu i točnost) u datoteku."""
    torch.save({"model_state": model.state_dict(), "class_names": CLASS_NAMES, **info}, path)


def load_model(path=MODEL_PATH, device="cpu"):
    """Napravi novi SimpleCNN i u njega učitaj spremljene parametre."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Datoteka '{path}' ne postoji. Najprije pokreni bilježnicu 1_treniranje.ipynb.")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Učitan model iz '{path}' (epoha {checkpoint['epoch']}, "
          f"točnost na validaciji {checkpoint['val_accuracy']:.1%})")
    return model


@torch.no_grad()
def predict_image(model, image, device, top_k=3):
    """Za jednu sliku vrati top_k najvjerojatnijih razreda kao listu parova (naziv, vjerojatnost)."""
    model.eval()
    x = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)  # [3, 64, 64] -> [1, 3, 64, 64]
    probabilities = torch.softmax(model(x), dim=1)[0]  # rezultati -> vjerojatnosti (zbroj je 1)
    top = probabilities.topk(top_k)
    return [(CLASS_NAMES[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
