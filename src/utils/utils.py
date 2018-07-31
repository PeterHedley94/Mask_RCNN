#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')


def print_progress(counter,total):
    inc = 5 #print every 10%
    percentage = str(counter/total * 100)
    title_string = percentage + str('% [')
    for i in range(round(counter/total * 100/inc)):
        title_string = title_string + '=='
    title_string = title_string + '>'
    for i in range(round(100/inc-round(counter/total * 100/inc))):
        title_string = title_string + '__'
    title_string = title_string + ']'
    title_string = title_string + "  Image " + str(counter) + " / " + str(total)
    print(title_string)
