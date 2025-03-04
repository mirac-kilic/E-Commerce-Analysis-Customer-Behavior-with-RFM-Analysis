#E-Commerce Sales Data Analysis: Customer Behavior and Sales Trends


# Import necessary libraries
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# Pandas settings: Show all columns, set display width
# and format float values to 3 decimal places
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x: '%.3f'%x)


# Load the dataset
df_= pd.read_csv('data.csv',encoding='ISO-8859-1')
df=df_.copy()
df1=df_.copy()

# First reviews
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.info())
print(df.columns)
print(df.describe().T)
print(df.nunique())


# Show the frequency and unique count of product descriptions
print(df['Description'].value_counts())
print(df['Description'].nunique())
print(df.isnull().sum())


# Analyze canceled orders and visualize the cancellation rate
df_cancellations = df1[df1['InvoiceNo'].str.contains('C', na=False)]
cancellation_rate = len(df_cancellations) / len(df1)
plt.bar(['Returned', 'Not Returned'], [len(df_cancellations), len(df1)-len(df_cancellations)], color=['red', 'green'])
plt.title(f'Return Rate: {cancellation_rate*100:.2f}%')
plt.ylabel('Number of Orders')
plt.show()

# Filter out canceled orders (InvoiceNo starting with 'C') and remove missing values
df=df[~df['InvoiceNo'].str.contains('C',na=False)]
df.dropna(inplace=True)


# Display the cleaned data and its statistics
print(df)
print(df.describe().T)


# Create a TotalPrice column (UnitPrice * Quantity)
df['TotalPrice']=df['UnitPrice']*df['Quantity']


# Top 10 Best Selling Products
category_sales = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)
category_sales.head(10).plot(kind='barh', color='lightcoral')
plt.title('Top 10 Best Selling Products')
plt.xlabel('Total Sales')
plt.show()
print(df)


# Show total sales quantities by product and total prices by invoice
print(df.groupby('Description').agg({'Quantity':'sum'}))
print(df.groupby('InvoiceNo').agg({'TotalPrice':'sum'}))




# Calculate RFM Metrics
print(df['InvoiceDate'].max())
today_date=dt.datetime(2011,12,11)
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])

# Calculate RFM metrics: Recency, Frequency, Monetary
rfm=df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (today_date-date.max()).days,
                                  'InvoiceNo': lambda invoice: invoice.nunique(),
                                  'TotalPrice': lambda total: total.sum()})
print(rfm.head(10))

# Rename the columns
rfm.columns=['recency','frequency','monetary']
print(rfm.describe().T)

# Calculate RFM Scores
rfm['R']=pd.qcut(rfm['recency'],q=5,labels=[5,4,3,2,1])
rfm['F']=pd.qcut(rfm['frequency'].rank(method='first'),q=5,labels=[1,2,3,4,5])
rfm['M']=pd.qcut(rfm['monetary'],q=5,labels=[1,2,3,4,5])

print(rfm)


# Combine RFM Scores
rfm['RFM_SCORE']=rfm['R'].astype(str)+rfm['F'].astype(str)
print(rfm)

# Show the best customers (RFM_SCORE = '55')
print(rfm[rfm['RFM_SCORE']=='55'])



# Define RFM Segments
seg_map={
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5":"cant_loose",
    r"3[1-2]":"about_to_sleep",
    r"33":"need_attention",
    r"[3-4][4-5]":"loyal_customers",
    r"41":"promising",
    r"51":"new_customers",
    r"[4-5][2-3]":"potential_loyalists",
    r"5[4-5]":"champions"
}

# Create RFM segments
rfm['segment']=rfm['RFM_SCORE'].replace(seg_map,regex=True)

print(rfm)


# RFM Segmentation Heatmap
segment_counts = rfm['segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Counts']

# Convert segments to integers
segment_counts['Counts'] = segment_counts['Counts'].astype(int)

# Creating a heat map
plt.figure(figsize=(8, 6))
sns.heatmap(segment_counts.set_index('Segment').T, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.5)
plt.title('RFM Segmentation Heatmap')
plt.xlabel('Segment')
plt.ylabel('Counts')
plt.show()


# Average RFM Metrics by Segment
segment_rfm = rfm.groupby('segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'
}).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=segment_rfm, x='segment', y='recency', hue='segment', palette='viridis', legend=False)
plt.title('Average Recency by Segment', fontsize=14)
plt.xlabel('Segment', fontsize=12)
plt.ylabel('Average Recency', fontsize=12)
plt.xticks(rotation=45)
plt.show()

 # RFM Metrics by Segment in Tabular Format
from tabulate import tabulate
print(tabulate(segment_rfm, headers='keys', tablefmt='pretty'))


# Segment Distribution
plt.figure(figsize=(8, 8))
segment_counts.set_index('Segment')['Counts'].plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Segment Distribution')
plt.ylabel('')
plt.show()