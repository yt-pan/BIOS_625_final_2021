#!/usr/bin/env python3

import csv
import os
import random

INPATH = ''
OUTPATH = ''

# 1 2 3 4 6 8 9 10 15 16 17 18 19 22==0 24==0
# removing inrel vars: each year each file
def main_sep():
    csv_files = os.listdir(INPATH)
    i = 0
    for csv_file in csv_files:
        i += 1
        f_in = INPATH + csv_file
        f_out = OUTPATH + csv_file
        print("begin ", csv_file, i, "/", len(csv_files))
        with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
            with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
                reader = csv.reader(csvfile)
                writer = csv.writer(opfile)
                for row in reader:
                    newrow = row[0:4] + [row[5]] + row[7:10] + row[14:19]
                    if row[0] == 'Year':
                        writer.writerow(newrow)
                    else:
                        if row[21] == '0' and row[23] == '0':
                            writer.writerow(newrow)

# removing inrel vars: merge every year
def main_merge():
    csv_files = os.listdir(INPATH)
    i = 0
    f_out = OUTPATH + 'Full.csv'
    allre = 0
    with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
        writer = csv.writer(opfile)
        for csv_file in csv_files:
            i += 1
            f_in = INPATH + csv_file
            print("begin ", csv_file, i, "/", len(csv_files))
            with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    newrow = row[0:4] + [row[5]] + row[7:10] + row[14:19]
                    if row[0] == 'Year':
                        if allre == 0:
                            writer.writerow(newrow)
                            allre = 1
                        else:
                            continue
                    else:
                        if row[21] == '0' and row[23] == '0':
                            writer.writerow(newrow)

# add catgorical depdelayC (0,1)
def add_cat_delay():
    csv_files = ["2003.csv", "2004.csv", "2005.csv", "2006.csv", "2007.csv", "2008.csv"]
    i = 0
    for csv_file in csv_files:
        i += 1
        f_in = INPATH + csv_file
        f_out = OUTPATH + csv_file
        print("begin ", csv_file, i, "/", len(csv_files))
        with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
            with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
                reader = csv.reader(csvfile)
                writer = csv.writer(opfile)
                for row in reader:
                    if row[0] == 'Year':
                        writer.writerow(row + ['depdelayC'])
                    else:
                        if int(row[9]) <= 15:
                            writer.writerow(row + ['0'])
                        else:
                            writer.writerow(row + ['1'])

# add catgorical depdelayC ('no-delay','delay')
def add_cat_delay_str():
    csv_files = ["2003.csv", "2004.csv", "2005.csv", "2006.csv", "2007.csv", "2008.csv"]
    i = 0
    for csv_file in csv_files:
        i += 1
        f_in = INPATH + csv_file
        f_out = OUTPATH + "03-08-str.csv"
        print("begin ", csv_file, i, "/", len(csv_files))
        with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
            writer = csv.writer(opfile)
            with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == 'Year':
                        writer.writerow(row + ['depdelayC'])
                    else:
                        if int(row[9]) <= 15:
                            writer.writerow(row + ['no-delay'])
                        else:
                            writer.writerow(row + ['delay'])

def merge():
    csv_files = ["2006.csv", "2007.csv", "2008.csv"]
    i = 0
    f_out ='D:\\' + '06-08.csv'
    allre = 0
    with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
        writer = csv.writer(opfile)
        for csv_file in csv_files:
            i += 1
            f_in = OUTPATH + csv_file
            print("begin ", csv_file, i, "/", len(csv_files))
            with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == 'Year':
                        if allre == 0:
                            writer.writerow(row)
                            allre = 1
                        else:
                            continue
                    else:
                        writer.writerow(row)

# Randomly select test set
def sel_testset():
    f_in = 'D:\\' + '06-08.csv'
    f_out1 = 'D:\\' + '68train.csv'
    f_out2 = 'D:\\' + '68test.csv'
    with open(f_out1, 'w', newline='', encoding='utf-8') as opfile:
        with open(f_out2, 'w', newline='', encoding='utf-8') as opfile2:
            writer = csv.writer(opfile)
            writer2 = csv.writer(opfile2)
            with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',' )
                for row in reader:
                    k = random.uniform(0,1)
                    if row[0] == 'Year':
                        writer.writerow(row)
                        writer2.writerow(row)
                    else:
                        if k<=0.7:
                            writer.writerow(row)
                        else:
                            writer2.writerow(row)       


if __name__ == '__main__':
    main_merge()
    main_sep()
    add_cat_delay()
    add_cat_delay_str()
    merge()
    sel_testset()
