#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:29:31 2018

@author: hanifa
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def cleanData(df):
    df=df[df['CustomerID'].notnull()]
    df=df[df['Quantity']>0]
    Quantity_min,Quantity_max=df['Quantity'].mean()-2* np.std(df['Quantity']),df['Quantity'].mean()+2* np.std(df['Quantity'])
    df=df[(df['Quantity']>=Quantity_min) & (df['Quantity']<=Quantity_max)]
    df['CustomerID']=df['CustomerID'].astype(int).astype(int)
    return df

def transformScaleData(df):
    df1=(df.groupby('CustomerID')
    .agg({'InvoiceNo': 'nunique',
          'StockCode': 'nunique',
          'Quantity':'sum',
          'UnitPrice':'mean'
         })
    .rename(columns={'InvoiceNo': 'NoOfInvoices', 
                     'StockCode': 'NoOfUniqueItems',
                     'Quantity':'TotalQuantity',
                     'UnitPrice':'UnitPriceMean'
                    })
    )

    df2=(df.groupby('CustomerID')
    .agg({
          'UnitPrice':'std'
         })
    .rename(columns={'UnitPrice':'UnitPriceStd'
                    })
    )

    df1=df1.join(df2,how='inner')

    df1['QuantityPerInvoice']=df1['TotalQuantity']/df1['NoOfInvoices']
    df1['UniqueItemsPerInvoice']= df1['NoOfUniqueItems']/df1['NoOfInvoices']
    df1.fillna({'UnitPriceStd':0},inplace=True)

    scaler = MinMaxScaler()
    df1_normalized=scaler.fit_transform(df1)
    
    df1_normalized_df= pd.DataFrame(data=df1_normalized,
                          index=df1.index,
                          columns=df1.columns)
    
    
    return df1,df1_normalized_df