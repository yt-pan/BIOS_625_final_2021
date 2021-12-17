#!/usr/bin/env python3

import csv
import random

INPATH = 'C:\\Users\\panyt\\OneDrive\\Coursework\\BIOS_625\\final\\cleaned_var_csv\\'
OUTPATH = 'E:\\BIOS625final\\add_catdelay_csv\\'

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
    sel_testset()
