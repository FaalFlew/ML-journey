# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
###################### ACCESSING DATA
#data = pd.Series([1,2,3])
import numpy as np

pd.set_option('display.width', 400)

pd.set_option('display.max_columns',13)


df= pd.read_csv('pokemon_data.csv')

#print(df)
#print(df.head(3))
#print(df.tail(3))

#row[0:4]   column[3:6]
#print(df.iloc[0:4][3:6])

#[row,column]
#print(df.iloc[2,1])


#print(df.iloc[8][3:6])
#print(df.iloc[3][3:9])

#print(df.columns)
#print(df['Name'])
#print(df['Name'][0:5])


#count is amount of rows, quartile example 25% meaning less than 25% of pokemons got 50hp or less.

#print(df.describe())
#print(df.sort_values(['HP', 'Name'],ascending=False))
#print(df.sort_values(['Type 1', 'HP'],ascending=[True,False]))



###################### CHANGING DATA
# create new column
#df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']

# remove a column
#df=df.drop(columns=['Total'])
#print(df.head(5))

# remove a column by index
#df=df.drop(df.columns[1], axis=1)

# Drop columns based on column index.
#df = df.drop(df.columns[[0, 1]],axis = 1)

#create new column equal to add all rows from column 4-9. axis=1 (1=adding horizontally, 0=vertically)
#df['total'] = df.iloc[0:800, 3:10].sum(axis=1)
#rename a column
#df =df.rename(columns={'total': 'Total'})
#print(df.tail(5))

#change order of a column
#cols = list(df.columns.values)
#[cols[-1]] (represents last column) is one column, sting is the return type, therefore wrap it in list []
#df = df[cols[0:4]+[cols[-1]]+ cols[4:12]]
#print(df.head(5))
#print(df.columns.get_loc("HP"))



###################### SAVING DATA

#save data with new column to new file
#df.to_csv('modified.csv')

#df.to_csv('modified.csv', index=False)
#df.to_csv('modified.txt',index=False,sep='\t')
#df.to_csv('pokemon_data.csv', index=False)

###################### DATA FILTERING

#df = df.loc[df['Type 1'] == 'Grass']
#df = df['Type 1'] == "Fire"
#print(df['Type 1'] == "Fire")
#print(df.loc[df['Type 1'] == "Fire"])
#print(df.loc[((df['Type 1'] == "Fire") | (df['Type 1'] == "Grass")) & (df['HP'] >100)])
#all rows that include the word "Mega"
#print(df.loc[(df['Name'].str.contains('Mega'))])
#print(df.loc[~(df['Name'].str.contains('Mega'))])

# using regex
import re
#print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)])
#print(df.loc[df['Name'].str.contains('^pi[a-z]', flags=re.I, regex=True)])


###################### CONDITIONAL CHANGES

# second 'Type 1' represents which column to change the value of. in this case we set column Type 1 = 'Flamer' if the value of column Type 1 = 'Fire'
#df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'
#here we set column Legendary = True if value of column Type 1 = 'Fire'
#df.loc[df['Type 1'] == 'Fire', ' Legendary'] = True

#change multiple columns conditionally
#df.loc[df['Total']>500, ['Generation','Legendary']] = 'TEST VALUE'


print(df)


#for index, row in df.iterrows():
    #print(index,row)

#for index, row in df.iterrows():
    #print(index,row['Name'])