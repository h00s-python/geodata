# EuroSAT – prepoznavanje satelitskih snimki neuronskom mrežom

Tri Jupyter bilježnice u kojima učenici korak po korak uče, testiraju i koriste konvolucijsku neuronsku mrežu (PyTorch) koja prepoznaje 10 vrsta terena na satelitskim snimkama Sentinel-2: šume, rijeke, ceste, naselja, polja...

## Datoteke

| Datoteka | Što radi |
|---|---|
| `1_treniranje.ipynb` | učitava podatke, trenira model, crta grafove učenja i sprema najbolji model u `eurosat_cnn.pt` |
| `2_testiranje.ipynb` | točnost na testnom skupu, točnost po razredima, matrica zabune, test otpornosti na iskvarene slike |
| `3_koristenje.ipynb` | predviđanje razreda za slike iz testnog skupa i za vlastite slike iz mape `moje_slike/` |
| `eurosat_common.py` | zajednički kod: nazivi razreda, priprema slika, model `SimpleCNN`, evaluacija, spremanje i učitavanje |

## Pokretanje

1. Kopiraj sve datoteke u istu mapu na JupyterHubu.
2. Otvori `1_treniranje.ipynb` i pokreni sve ćelije redom (**Shift + Enter**). Prva ćelija instalira pakete `datasets` i `matplotlib`, a prvo učitavanje podataka preuzima oko 100 MB s Hugging Facea.
3. Kad se treniranje završi, pokreni `2_testiranje.ipynb`, pa `3_koristenje.ipynb`.

Prva ćelija s kodom ispisuje verziju PyTorcha i naziv grafičke kartice. Ako se tu pojavi **UPOZORENJE**, pogledaj odjeljak *Rješavanje problema*.

## Postavke

Postavke treniranja nalaze se u koraku 2 bilježnice `1_treniranje.ipynb`:

| Postavka | Zadano | Značenje |
|---|---|---|
| `EPOCHS` | 20 | koliko puta model prođe kroz sve slike za treniranje |
| `BATCH_SIZE` | 128 | broj slika u jednoj seriji |
| `LEARNING_RATE` | 0.001 | stopa učenja |
| `VALIDATION_FRACTION` | 0.1 | udio slika za treniranje koji se odvaja za validaciju |
| `NUM_WORKERS` | 2 | broj pomoćnih procesa koji pripremaju slike (0 = bez pomoćnih procesa) |
| `SEED` | 42 | sjeme slučajnosti, za ponovljive rezultate |

Na kraju bilježnice nalaze se prijedlozi pokusa (*Za istraživanje*).

## Model

`SimpleCNN` (u `eurosat_common.py`) je jednostavna konvolucijska mreža s oko 1,2 milijuna parametara:

- 4 konvolucijska bloka (Conv → BatchNorm → ReLU, dvaput, pa MaxPool) s 32, 64, 128 i 256 mapa značajki; slika se smanjuje 64 → 32 → 16 → 8 → 4 piksela,
- zatim prosjek po svakoj mapi značajki, Dropout i jedan potpuno povezani sloj s 10 izlaza.

Treniranje koristi augmentaciju (zrcaljenje, svjetlina, kontrast), optimizator Adam, kosinusni raspored stope učenja i mješovitu preciznost (bfloat16) na grafičkoj kartici. Sprema se model iz epohe s najboljom točnošću na validacijskom skupu.

## Skup podataka

Hugging Face: [`tanganke/eurosat`](https://huggingface.co/datasets/tanganke/eurosat) – slike 64 × 64 piksela (RGB).

> Na opisu skupa piše `load_dataset('tranganke/eurosat')`, ali to je tipfeler: to ime ne postoji. Ispravno je **`tanganke/eurosat`**.

| Oznaka | Razred (engleski) | Razred u bilježnicama |
|---|---|---|
| 0 | annual crop land | jednogodišnji usjevi |
| 1 | forest | šuma |
| 2 | brushland or shrubland | grmlje i šikara |
| 3 | highway or road | autocesta ili cesta |
| 4 | industrial buildings or commercial buildings | industrijske ili poslovne zgrade |
| 5 | pasture land | pašnjak |
| 6 | permanent crop land | trajni nasadi |
| 7 | residential buildings or homes or apartments | stambene zgrade |
| 8 | river | rijeka |
| 9 | lake or sea | jezero ili more |

Dijelovi skupa: `train` (21 600 slika), `test` (2 700) te 7 iskvarenih verzija testnog skupa (po 2 700): `contrast`, `gaussian_noise`, `impulse_noise`, `jpeg_compression`, `motion_blur`, `pixelate`, `spatter`. Skup nema validacijski dio, pa bilježnica 1 odvaja 10 % slika za treniranje.

## Rješavanje problema

**UPOZORENJE: ovaj PyTorch nije napravljen za ovu karticu** (ili greška *no kernel image is available*)
RTX 5090 (arhitektura Blackwell, `sm_120`) treba PyTorch 2.7 ili noviji, izgrađen za CUDA 12.8. U ćeliji ili terminalu pokreni:

```
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

i zatim ponovno pokreni kernel (*Kernel → Restart Kernel*).

**Greška *bus error*, *shared memory* ili *DataLoader worker exited unexpectedly***
Pomoćni procesi predaju pripremljene serije slika kroz dijeljenu memoriju (`/dev/shm`), a u JupyterHub kontejnerima ona često ima samo 64 MB. Sa zadanim postavkama (`NUM_WORKERS = 2`, `BATCH_SIZE = 128`) izmjereno je najviše 36 MB. Ako i dalje dobivaš grešku, postavi `NUM_WORKERS = 0` (u bilježnici 1 u koraku 2, u bilježnici 2 u koraku 0) – priprema slika bit će otprilike dvaput sporija. Ako si povećao/la `BATCH_SIZE`, smanji ga ili postavi `NUM_WORKERS = 0`.

Ako je greška prekinula ćeliju, ponovno pokreni kernel (*Kernel → Restart Kernel*) prije nastavka.

**`FileNotFoundError: Datoteka 'eurosat_cnn.pt' ne postoji`**
Najprije do kraja pokreni `1_treniranje.ipynb`. Sve datoteke moraju biti u istoj mapi.

**`ModuleNotFoundError: No module named 'datasets'`**
Pokreni prvu ćeliju s kodom u `1_treniranje.ipynb` (`%pip install ...`).

**Preuzimanje skupa podataka ne radi**
Računalo treba pristup internetu (huggingface.co). Ako ide preko proxyja, postavi varijable okruženja `HTTPS_PROXY`/`HTTP_PROXY`.

## Literatura

Skup podataka EuroSAT:

```bibtex
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={12},
  number={7},
  pages={2217--2226},
  year={2019},
  publisher={IEEE}
}
```

Radovi uz koje je objavljena verzija skupa na Hugging Faceu (s iskvarenim testnim skupovima):

```bibtex
@misc{yangAdaMergingAdaptiveModel2023,
  title = {{{AdaMerging}}: {{Adaptive Model Merging}} for {{Multi-Task Learning}}},
  shorttitle = {{{AdaMerging}}},
  author = {Yang, Enneng and Wang, Zhenyi and Shen, Li and Liu, Shiwei and Guo, Guibing and Wang, Xingwei and Tao, Dacheng},
  year = {2023},
  month = oct,
  number = {arXiv:2310.02575},
  eprint = {2310.02575},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2310.02575},
  url = {http://arxiv.org/abs/2310.02575},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}

@misc{tangConcreteSubspaceLearning2023,
  title = {Concrete {{Subspace Learning}} Based {{Interference Elimination}} for {{Multi-task Model Fusion}}},
  author = {Tang, Anke and Shen, Li and Luo, Yong and Ding, Liang and Hu, Han and Du, Bo and Tao, Dacheng},
  year = {2023},
  month = dec,
  number = {arXiv:2312.06173},
  eprint = {2312.06173},
  publisher = {arXiv},
  url = {http://arxiv.org/abs/2312.06173},
  archiveprefix = {arxiv},
  copyright = {All rights reserved},
  keywords = {Computer Science - Machine Learning}
}

@misc{tangMergingMultiTaskModels2024,
  title = {Merging {{Multi-Task Models}} via {{Weight-Ensembling Mixture}} of {{Experts}}},
  author = {Tang, Anke and Shen, Li and Luo, Yong and Yin, Nan and Zhang, Lefei and Tao, Dacheng},
  year = {2024},
  month = feb,
  number = {arXiv:2402.00433},
  eprint = {2402.00433},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2402.00433},
  url = {http://arxiv.org/abs/2402.00433},
  archiveprefix = {arxiv},
  copyright = {All rights reserved},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}
```
