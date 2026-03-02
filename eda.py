import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('online_shoppers_intention.csv')

print("=== INFO DATASET ===")
print(df.shape)
print(df.dtypes)
print(df.describe())

# ================================
# 1. Distribusi Revenue
# ================================
plt.figure(figsize=(6,4))
df['Revenue'].value_counts().plot(kind='bar', color=['salmon','steelblue'])
plt.title('Distribusi Revenue (Beli vs Tidak)')
plt.xlabel('Revenue')
plt.ylabel('Jumlah')
plt.xticks(ticks=[0,1], labels=['Tidak Beli', 'Beli'], rotation=0)
plt.tight_layout()
plt.savefig('chart_revenue.png')
plt.show()

# ================================
# 2. Transaksi per Bulan
# ================================
month_order = ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly = df[df['Revenue']==True]['Month'].value_counts().reindex(month_order)

plt.figure(figsize=(10,4))
monthly.plot(kind='bar', color='steelblue')
plt.title('Jumlah Transaksi per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Beli')
plt.tight_layout()
plt.savefig('chart_monthly.png')
plt.show()

# ================================
# 3. Visitor Type vs Revenue
# ================================
plt.figure(figsize=(8,4))
sns.countplot(data=df, x='VisitorType', hue='Revenue', palette='Set2')
plt.title('Visitor Type vs Revenue')
plt.tight_layout()
plt.savefig('chart_visitor.png')
plt.show()

# ================================
# 4. Korelasi Heatmap
# ================================
plt.figure(figsize=(12,8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelasi Antar Fitur')
plt.tight_layout()
plt.savefig('chart_correlation.png')
plt.show()

print("=== EDA Selesai! Chart tersimpan ===")