# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
# Importing the libraries
import numpy as np
import numpy.linalg as lg
import csv


######  Read in files into 2 numpy matrices
def readDataFile(fileName):
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    # loadtxt defaults to floats, use dtype to specify string
    # usecols chooses the columns to use, by default, all columns are used.
    # skiprows skips a header row if you need to
    data = np.loadtxt(raw_data, skiprows = 1, delimiter=",", dtype = 'str')
    names = data[:, 0]
    stats = data[:,1:6].astype(int)
    
    return names, stats


######  Display the names and stats
def printStats(desc, names, stats):
    print('\n', desc, ':')
    for index in range (len(names)):
        #print(names_1998[index], end='')
        print('{:>18}'.format(names[index]), ': ', end='')
        for col in range ( len(stats[0])):
            print(stats[index][col], ' ', end='')
        print('')   # newline
        
        
        
# =============================================================================
# Read in the matrix1998 file
# =============================================================================
names_1998, stats_1998 = readDataFile('Football_matrices_1998.csv')

# =============================================================================
# Read in the matrix1999 file
# =============================================================================
names_1999, stats_1999 = readDataFile('Football_matrices_1999.csv')



# =============================================================================
# create a new stats matrix with the sum of stats and display
# =============================================================================

sum = stats_1998+stats_1999
print(sum)

# =============================================================================
# Find the Difference and print
# =============================================================================

diff = stats_1999-stats_1998
print(diff)
# =============================================================================
# Write a method to compute the QB ratings given a matrix of stats
#         - could be 1998, 1999, or the sum of both years
# =============================================================================
def getRatings(stats):
    ratings = ((((stats[:,1]/stats[:, 0])-.3)*5 +
         ((stats[:,2]/stats[:,0]-3)*.25) +
         ((stats[:,3]/stats[:,0])*20) +
         (2.375-(stats[:,4]/stats[:,0])*25))/6)*100
    return ratings


# =============================================================================
# Use the method above to get the QB ratings for 1998 and 1999
# then display the rating for each QB for both years
#
# expected output:
# QB Ratings:
#       Quarterback :   1998    1999
#       Troy Aikman :  88.46   81.12
#        Tony Banks :  68.62   81.20
#        Jeff Blake :  78.20   77.60
#   Steve Beuerlein :  88.25   94.58
#  
# =============================================================================

ratings_1998 = getRatings(stats_1998)
ratings_1999 = getRatings(stats_1999)



print('\n', 'QB Ratings:')
print('{:>18}'.format("Quarterback"), ': ', end='')
print('{:>6}'.format('1998'), ' ' , end='')
print('{:>6}'.format('1999'))


for index in range(len(names_1999)):
    print('{:>18}'.format(names_1999[index]), ': ', end='')
    print('{:6.2f}'.format(ratings_1998[index]), ' ' , end='')
    print('{:6.2f}'.format(ratings_1999[index]))





