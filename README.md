# Laporan Machine Learning
### Nama  : Rohmat
### NIM   : 211351131
### Kelas : TIF Pagi B

## Domain Proyek


## Business Understanding


### Problem Statments


### Goals

  
### Solution statements


## Data Understanding
Dataset yang digunakan adalah Grocery Store Data Set, yang berisi sekumpulan data transaksi di sebuat toko

[Grocery Store Data Set ('https://www.kaggle.com/datasets/shazadudwadia/supermarket')]

Berikut adalah items-items yang ada di dalam dataset Grocery Store Data Set
- JAM (Selai)
- MAGGI (Bumbu penyedap Maggi)
- SUGAR (Gula)
- COFFEE (Kopi)
- COCK (Ayam)
- TEA (Teh)
- BOURNVITA (Minuman Bournvita)
- CORNFLAKES (Sereal Jagung)
- BREAD (Roti)
- BISCUIT (Biskuit)
- MILK (Susu)
Semua Bertipe data Object

## Data Preperation

### Import Dataset

Pertama kita harus terhubung dulu dengan kaggle, dengan cara mendownload Token dari kaggle terlebih dahulu, setelah mendapatkan token kita bisa langsung memasukan peritah di bawah ini dan masukan token yang sudah di download.
```bash
from google.colab import files
files.upload()
```
Langkah selanjutnya adalah membuat direktori
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Setelah direktori di buat kita bisa langsung memanggil url dataset dari kaggle tersebut dengan cara mengcopy API command nya
```bash
!kaggle datasets download -d shazadudwadia/supermarket
```
Setelah dataset nya berhasil terpanggil kita harus meng ekstrak dataset tersebut dengan cara memasukan perintah seperti berikut ini :
```bash
! mkdir mobile-price
! unzip mobile-price.zip -d mobile-price
! ls mobile-phone-specifications-and-prices
```
Nah pada tahap ini kita sudah melakukan import dataset dari kaggle.

### Import library
Setelah melakukan import dataset kita dapat melanjutkan ke tahap berikutnya yaitu mengimport library yang akan kita gunakan 
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pandas.plotting import parallel_coordinates
```

### Data Discovery

