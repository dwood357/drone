# -*- coding: utf-8 -*-
"""
Created on Tue Sept 19 10:57:47 2017

@author: Daniel Wood
"""
import serial
import numpy as np

# from pyqtgraph.Qt import QtGui
# import pyqtgraph as pg

ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM4'
ser.open()

f = open('nmea_sentences.txt', 'w+')

while True:

        #this is a blocking function
        nmea_string = ser.readline().decode('ascii', errors='replace')

        # print(nmea_string)
        

        if (nmea_string[0:6] == '$GNRMC'):

                print(nmea_string)
                data = nmea_string.split(',')
                

                UTC = data[1]
                latitude = data[3]
                lat_dir = data[4]
                longitude = data[5]
                long_dir = data[6]
                Speed = data[7]
                date = data[9]


                #divide by 100 to use in the distance calc.
                latitude = float(latitude)/100
                longitude = float(longitude)/100

                date = str(date)
                date = [date[i:i+2] for i in range(0,len(date),2)]

                day = date[0]
                month = date[1]
                year = '20' + date[2]

                # date_data = np.array([UTC, month, day, year])
                str_data = np.array([UTC, latitude, lat_dir, longitude, long_dir, Speed, day, month, year])

                # # print(str(self.str_data))

                f.write(str(str_data))
                f.write('\n')
                f.flush()

# f.close()