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


## Import Dataset

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

## Import library
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

## Data Discovery
Langkah pertama adalah memanggil dataset yang akan di gunakan , dan langsung memunculkan isi dataset nya
```bash
df=pd.read_csv('/content/supermarket/GroceryStoreDataSet.csv')
df
```
Setelah itu kita akan menggunakan kode di bawah ini untuk menampilkan statistik deskriptif isi dari dataset kita
```bash
df.describe()
```
Lalu kita cek tipe datanya
```bash
df.info()
```
Selanjutnya kita akan memberikan nama pada kolom tabel 
```bash
df.columns=['NAME ITEMS']
```
Lalu kita bisa cek kembali dataset kita
```bash
df
```

## Exploratory Data Analysis (EDA)
Pada tahap eksploratory data analysis (EDA) langkah awal kita bisa mengecek korelasi antar variabel
```bash
df = pd.DataFrame(data)
transactions = df['NAME ITEMS'].str.split(',')
one_hot_df = pd.get_dummies(transactions.apply(pd.Series).stack()).sum(level=0)
plt.figure(figsize=(12, 8))
sns.heatmap(one_hot_df.corr(), annot=True, cmap="coolwarm")
plt.title('Sebaran Items (Heatmap)')
plt.show()
```
![image](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/5000a87b-b3ac-4910-a61c-e7a37da8a569)

Lalu kita bisa melihat Frequencies kemunculan item yang paling banyak dalam data transaksi
```bash
df = pd.DataFrame(data)
df['items_list'] = df['NAME ITEMS'].apply(lambda x: x.split(','))
all_items = [item for sublist in df['items_list'] for item in sublist]
item_counts = pd.Series(all_items).value_counts()
colors = sns.color_palette('husl', len(item_counts))
plt.figure(figsize=(12, 8))
item_counts.plot(kind='bar', color=colors)
plt.title('Items Frequencies', fontsize=13, fontweight='bold')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
```
![image](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/ea461d36-7ac1-422d-adb8-1a63e3922a80)

Selanjutnya kita akan lakukan perbandingan kemunculan item dengan persentase 
```bash
df = pd.DataFrame(data)
items_to_compare = ['JAM', 'MAGGI', 'SUGER', 'COFFEE', 'CHEESE', 'TEA', 'BOURNVITA', 'CORNFLAKES', 'BREAD', 'BISCUIT', 'MILK']
item_counts = df['NAME ITEMS'].apply(lambda x: pd.Series(x.split(','))).stack().value_counts()
item_counts = item_counts[item_counts.index.isin(items_to_compare)]
colors = sns.color_palette('husl', len(item_counts))
plt.figure(figsize=(12, 6))
ax = item_counts.sort_values().plot(kind='barh', color=colors)
plt.title('Perbandingan Jumlah Kemunculan Items dalam Transaksi', fontsize=12, fontweight='bold')
plt.xlabel('Jumlah Kemunculan')
plt.ylabel('Item')
for bar in ax.patches:
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2 - 0.15,
             f'{(bar.get_width() / len(df) * 100):.2f}%', ha='center', va='bottom', color='black')

plt.show()
```
![image](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/72773977-a000-46db-89cb-d01d47591920)

Lalu kita akan bandingkan menggunakan Pie plot , produk mana yang sering di beli pelanggan

```bash
df = pd.DataFrame(data)
item_totals = df['NAME ITEMS'].str.split(',', expand=True).stack().value_counts()

# Pie plot perbandingan pembelian
plt.figure(figsize=(8, 8))
plt.pie(item_totals, labels=item_totals.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Perbandingan Pembelian Setiap Item', fontsize=13, fontweight='bold')
plt.show()
```
![image](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/931aad73-4b9f-49b5-8a59-0f38c22d5be4)

kita dapat melihat item paling banyak terjual dengan tree map
```bash
df = pd.DataFrame(data)
item_totals = df['NAME ITEMS'].str.split(',', expand=True).stack().value_counts().reset_index()
item_totals.columns = ['itemDescription', 'count']

# Membuat treemap
fig = px.treemap(item_totals, path=['itemDescription'], values='count', title='Frequency Items yang terjual')
fig.update_layout(title_x=0.5, title_font=dict(size=18))
fig.update_traces(textinfo="label+value",textfont=dict(color='white'))
# Menampilkan treemap
fig.show()
```
![Screenshot 2024-01-02 131745](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/921a8bf5-8c38-42da-9930-ea84e3af3b52)


## Preprocessing

```bash
path=('supermarket/GroceryStoreDataSet.csv')
df = pd.read_csv(path,names = ['items'], sep = ',')
df.head(5)
```

```bash
df_1  = df['items'].apply(lambda x : x.split(','))
df_1.head(3)
```

```bash
liste = []

for i in range(len(df_1)):
    x = df_1[i]
    print(x)
    liste.append(x)
```


```bash
encoder = TransactionEncoder()
pred = encoder.fit_transform(liste)
df = pd.DataFrame(pred,columns = encoder.columns_)
df.head(5)
```


## modeling


```bash
frequency = apriori(df,min_support = 0.15,use_colnames = True,verbose = 1)
frequency
```

```bash
rules =  association_rules(frequency,metric = 'lift',min_threshold = 0.8)
rules
```

```bash
rules['antecedent'] = rules['antecedents'].apply(lambda antecedents : list(antecedents)[0])
rules['consequent'] = rules['consequents'].apply(lambda consequents : list(consequents)[0])
rules['rule'] = rules.index
rules
```

```bash
rules.sort_values('lift',ascending = False).head(5)
```

```bash
rules.sort_values('confidence',ascending = False).head(5)
```


## Visualisasi Hasil Algoritma

```bash
fig, ax = plt.subplots(figsize = (30,30))
pivot = rules.pivot(index = 'consequents',
                   columns = 'antecedents', values= 'lift')
sns.heatmap(pivot, annot=True, cmap='YlGnBu', cbar=True, fmt='.2f', annot_kws={"size": 14},
            cbar_kws={'label': 'Lift Value'})
plt.show()
```
![image](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/8f53da97-34eb-43af-b704-966fe86d1f0f)


```bash
fig = px.scatter(x=rules['support'], y=rules['confidence'])
fig.show()
```


```bash
cords = rules[['antecedent','consequent','rule']]

parallel_coordinates(cords,'rule',colormap = 'ocean')
```
![image](https://github.com/RohmatIF/Rohmat_ML_Apriori_UAS/assets/147891420/734c5b22-3732-4e05-b53f-f133fa2f7585)


