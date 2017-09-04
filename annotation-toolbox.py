#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from tkinter import *

from drawing import Map

# Inizializza la mappa
tkmaster = Tk(className="annotation-toolbox")
map = Map(tkmaster, scale=18, width=400, height=400, bg="#FFFFFF")
map.pack(expand=True, fill="both", side="right")

global args


########################################################################################################################
#                                                   LEGEND
########################################################################################################################
def show_legend():
    map.clear_log()
    map.log(txt="0:\tDataset loading\n")
    map.log(txt="L:\tShow legend\n")
    map.log(txt="1:\tPositive\n")
    map.log(txt="2:\tNegative\n")
    map.log(txt="3:\tNeutral\n")
    map.log(txt="4:\tSkip\n")
    map.log(txt="Esc:\tExit\n")


########################################################################################################################

########################################################################################################################
#                                                   FUNCTIONS                                                          #
########################################################################################################################

def load_dataset(event):
    map.clear_log()
    map.log(txt=">> 1: Dataset loading\n\n")
    global images, dataset
    images = [k for k in sorted(os.listdir(os.path.join(args.images))) if '.jpg' in k]
    dataset = []
    try:
        with open(args.data, 'rb') as features:
            data = features.readlines()
            for i, line in enumerate(data):
                if (i != 0):
                    dataset.append(line.decode().split(',')[-2].split('/')[-1])
        features.close()
    except FileNotFoundError:
        map.log(txt="Wrong data file or file path!\n")
        dataset = []
        f = open(args.data, 'a')
        f.write('images' + ',' + 'target' + '\n')
        f.close()

    images = [x for x in images if x not in dataset]
    map.log(txt="Data Loaded.\n")

    show_image()


def show_image():
    map.log(txt=">> {0}\n".format(images[0]))
    map.draw_image(os.path.join(args.images, images[0]))


def legend(event):
    show_legend()
    pass


def positive(event):
    f = open(args.data, 'a')
    f.write(os.path.join(args.images, images[0]) + ',' + 'positive' + '\n')
    f.close()

    del images[0]
    map.clear_log()
    map.log(txt=">> 1: {0}: Positive\n\n".format(images[0]))
    show_image()


def negative(event):
    f = open(args.data, 'a')
    f.write(os.path.join(args.images, images[0]) + ',' + 'negative' + '\n')
    f.close()

    del images[0]
    map.clear_log()
    map.log(txt=">> 2: {0}: Negative\n\n".format(images[0]))
    show_image()


def neutral(event):
    f = open(args.data, 'a')
    f.write(os.path.join(args.images, images[0]) + ',' + 'neutral' + '\n')
    f.close()

    del images[0]
    map.clear_log()
    map.log(txt=">> 3: {0}: Neutral\n\n".format(images[0]))
    show_image()


def skip(event):
    f = open(args.data, 'a')
    f.write(os.path.join(args.images, images[0]) + ',' + 'skip' + '\n')
    f.close()

    del images[0]
    map.clear_log()
    map.log(txt=">> 4: {0}: Skip\n\n".format(images[0]))
    show_image()

def close(event):
    tkmaster.withdraw()  # if you want to bring it back
    sys.exit()  # if you want to exit the entire thing


########################################################################################################################

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('-images', dest='images', action='store', default='images', help='images path file *.png')
    p.add_argument('-data', dest='data', action='store', default='data.csv', help='data path file *.csv')

    global args
    args = p.parse_args()

    show_legend()

    tkmaster.bind("0", load_dataset)
    tkmaster.bind("l", legend)
    tkmaster.bind("1", positive)
    tkmaster.bind("2", negative)
    tkmaster.bind("3", neutral)
    tkmaster.bind("4", skip)
    tkmaster.bind('<Escape>', close)

    mainloop()


if __name__ == '__main__':
    main()
