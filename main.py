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
#import re
#print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)])
#print(df.loc[df['Name'].str.contains('^pi[a-z]', flags=re.I, regex=True)])


###################### CONDITIONAL CHANGES

# second 'Type 1' represents which column to change the value of. in this case we set column Type 1 = 'Flamer' if the value of column Type 1 = 'Fire'
#df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'
#here we set column Legendary = True if value of column Type 1 = 'Fire'
#df.loc[df['Type 1'] == 'Fire', ' Legendary'] = True

#change multiple columns conditionally
#df.loc[df['Total']>500, ['Generation','Legendary']] = 'TEST VALUE'


#print(df)


#for index, row in df.iterrows():
    #print(index,row)

#for index, row in df.iterrows():
    #print(index,row['Name'])

import sys
#print("Python version: {}".format(sys.version))
import pandas as pd
#print("pandas version: {}".format(pd.__version__))
import matplotlib
#print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
#print("NumPy version: {}".format(np.__version__))
import scipy as sp
#print("SciPy version: {}".format(sp.__version__))
import IPython
#print("IPython version: {}".format(IPython.__version__))
import sklearn
#print("scikit-learn version: {}".format(sklearn.__version__))

# create a simple dataset of people
#data = {'Name': ["John", "Anna", "Peter", "Linda"],
 #'Location' : ["New York", "Paris", "Berlin", "London"],
 #'Age' : [24, 13, 53, 33]
 #}
#data_pandas = pd.DataFrame(data)
#print(data_pandas)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
#display(data_pandas)




###################### NUMPY

#(rows, cols)
#print(np.array([1,2,3]))
#print(np.ones(6))
#print(np.ones((6), dtype=str))
#print(np.zeros((6,5), dtype=int))
#print(np.arange(1,51))
#print(np.arange(1,51).reshape(2,5,5))

#print(np.ones((3,4)))

#print(np.empty((2,4), dtype=int))

#print(np.linspace(0, 1, 6, endpoint=True))

#a = np.array([1,2,3])
#a2 = np.array([3,2,1])

#b = np.array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
#c = np.array([[[1,2,3],[1,2,3],[2,3,4],[2,3,4]]])
#d = np.array([[[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
#              [[5, 6, 7], [5, 6, 7], [8, 9, 10], [8, 9, 10]]])

#print(a2*a)
#print(a.ndim)
#cols for 1 d,   [rows,cols] for 2d, nests X firstaxis X secondaxis for 3d
#print("the shape is {}".format(b.shape))
#print(f'{"dd"}{b.shape}')

#print(c.shape)
#print(d.shape)

#get specific element [row,column]

# Get a specific element [r,c]
#print(d[1,:,1])
#More advanced [startindex:endindex:stepsize]
#print(b[0,1:6:2])

#change list items
#b[1,5] = 20
#print(b)
#b[:,2] = 5
#print(b)
#b[:,2] = [1,2]
#print(b)
#d[:,1,:] = [[9,9,9],[8,8,8]]
#print(d)

#print(np.full(d.shape,4))
#print(np.full_like(d,4))
#print(a.shape)
#print(d.shape)

#print(np.random.rand(4,2))
#print(np.random.random_sample(d.shape))

#random numbers but dont give me higher than 2,
#print(np.random.randint(2,size=(3,3)))
#random numbers between 2,6 excluding 2,6
#print(np.random.randint(2,6,size=(3,3)))
#identity matrix
#print(np.identity(3))

#arr = np.array([[1,2,3]])
#r1 = np.repeat(arr,4,axis=0)
#print(r1)



#f = np.ones((5,5))
#f[1:4,1:4] = np.zeros((3,3))
#f[2,2] = 9
#print(f)

#k = f.copy()
#print(k)

#print(a + 2)
#print(np.sin(a))

#Multiply matrices
#a = np.ones((2,3))
#b = np.full((3,2),2)
#print(np.matmul(a,b))

# Find determinant
#c= np.identity(3)
#print(np.linalg.det(c))

#Statistics
#a = np.array([[1,2,3],
#              [4,5,6]])
#print(np.min(a,axis=1))
#print(np.max(a,axis=1))
#print(np.sum(a,axis=1))
#print(np.sum(a,axis=0))

# Reorganizing arrays
#a = np.array([[1,2,3,4],[5,6,7,8]])
#print(a)
#print(f'{a.shape}\n-----')
#b = a.reshape((4,2))
#print(f'{b}\n-----')
#b = a.reshape((2,2,2))
#print(f'{b}\n-----')

#Vertically and horizontally stacking vectors
#a = np.array([1,2,3,4])
#b = np.array([5,6,7,8])
#print(np.array([a,b,a,b]))
#print(np.array((a,b,a,b)))
#print(np.vstack([a,b,a,b]))
#print(np.vstack((a,b,a,b)))
#print(f'{np.hstack([a,b])}\n-----')
#print(f'{np.hstack((a,b))}\n-----')

#a = np.ones((2,4))
#b = np.zeros((2,2))
#print(np.hstack(a,b))



# Matplotlib

import matplotlib.pyplot as plt

x = [0,1,2,3,4]
y = [0,2,4,6,8]

# Resize your Graph (dpi specifies pixels per inch. When saving probably should use 300 if possible)
plt.figure(figsize=(8,5), dpi=100)

# Line 1

# Keyword Argument Notation
#plt.plot(x,y, label='2x', color='red', linewidth=2, marker='.', linestyle='--', markersize=10, markeredgecolor='blue')

# Shorthand notation
# fmt = '[color][marker][line]'
plt.plot(x,y, 'b^--', label='2x')

## Line 2

# select interval we want to plot points at
x2 = np.arange(0,4.5,0.5)

# Plot part of the graph as line
plt.plot(x2[:6], x2[:6]**2, 'r', label='X^2')

# Plot remainder of graph as a dot
plt.plot(x2[5:], x2[5:]**2, 'r--')

# Add a title (specify font parameters with fontdict)
plt.title('Measuring nothing', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})

# X and Y labels
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# X, Y axis Tickmarks (scale of your graph)
plt.xticks([0,1,2,3,4,])
#plt.yticks([0,2,4,6,8,10])

# Add a legend
plt.legend()

# Save figure (dpi 300 is good when saving so graph has high resolution)
plt.savefig('mygraph.png', dpi=300)

# Show plot
plt.show()






