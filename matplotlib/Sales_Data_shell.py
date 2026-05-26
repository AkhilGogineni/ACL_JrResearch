# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the libraries
import numpy as np
import numpy.linalg as lg
import csv
import numpy.random as random
import matplotlib.pyplot as plt

# Importing the dataset


def createData():
    fileName = 'company_sales_data.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    # loadtxt defaults to floats
    data = np.loadtxt(raw_data,  delimiter=",", dtype='str')
    header = data[0:1, :]
    rows = data[1:, :].astype('float')

    return header, rows



def ex1_CompanyProfit(header, rows):
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]

    ax.plot(rows[:,0], rows[:,(len(rows[0])-1)], color ="blue")
    plt.ylim(100000)
    plt.xticks(np.arange(0, len(rows)+1, 1))
    ax.set_xlabel("Month Number")
    ax.set_ylabel("Total Profit")
    ax.set_title('Company profit per month')
    
    pass 

def ex2_CompanyProfit_style(header, rows):
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]

    ax.plot(rows[:,0], rows[:,(len(rows[0])-1)], color ="red", linestyle = "dotted", marker = "o", mfc = "black", linewidth = 3, label = "Profit Data of Last Year")
    plt.ylim(100000)
    plt.xticks(np.arange(0, len(rows)+1, 1))
    ax.set_xlabel("Month Number")
    ax.set_ylabel("Total Profit")
    ax.set_title('Company profit per month')
    plt.legend(loc="lower right")
    pass


def ex3_CompanyProfit_style(header, rows):
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]

    ax.plot(rows[:,0], rows[:,1], color ="blue", marker = "o", linewidth = 3, label = "Face Cream Sales Data")
    ax.plot(rows[:,0], rows[:,2], color ="orange", marker = "o", linewidth = 3, label = "Face Wash Sales Data")
    ax.plot(rows[:,0], rows[:,3], color ="red", marker = "o", linewidth = 3, label = "Toothpaste Sales Data")
    ax.plot(rows[:,0], rows[:,4], color ="purple", marker = "o", linewidth = 3, label = "Bathing Soap Sales Data")
    ax.plot(rows[:,0], rows[:,5], color ="green", marker = "o", linewidth = 3, label = "Shampoo Sales Data")
    ax.plot(rows[:,0], rows[:,6], color ="brown", marker = "o", linewidth = 3, label = "Moisturizer Sales Data")

    plt.ylim(1000)
    plt.xticks(np.arange(0, len(rows)+1, 1))
    ax.set_xlabel("Month Number")
    ax.set_ylabel("Sales Units in Number")
    ax.set_title('Company profit per month')
    plt.legend(loc="upper left")
    pass


def ex4_ToothPaste_grid(header, rows):
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.15,0.8,0.7])# [left, bottom, width, height]

    ax.scatter(rows[:,0], rows[:,3], color ="blue", marker = "o", label = "Toothpaste Sales Data")
    
    plt.grid(linestyle = "--")
    plt.ylim(4500)
    plt.xticks(np.arange(0, len(rows)+1, 1))
    ax.set_xlabel("Month Number")
    ax.set_ylabel("Sales Units in Number")
    ax.set_title('Company profit per month')
    plt.legend(loc="upper left")
    pass


def ex5_BarChart(header, rows):
    
    fig = plt.figure()
    
    ax = fig.add_axes([0.1,0.15,0.8,0.7])
    plt.xticks(np.arange(0, len(rows)+1, 1))
    plt.title("Facecream and Facewash Sales Data")
    ax.set_ylabel("Sales units in Number")
    ax.set_xlabel("Month Number")
    ax.bar(rows[:,0]-0.1, rows[:,1], color ="blue",  label = "Face Cream Sales Data", width = 0.2)
    ax.bar(rows[:,0]+0.1, rows[:,2], color ="orange",   label = "Face Wash Sales Data", width = 0.2)
    plt.legend(loc="upper left")
    plt.grid(linestyle = '--')
    pass


header, rows = createData()
ex1_CompanyProfit(header, rows)
ex2_CompanyProfit_style(header, rows)
ex3_CompanyProfit_style(header, rows)
ex4_ToothPaste_grid(header, rows)
ex5_BarChart(header, rows)