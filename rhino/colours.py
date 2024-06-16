#!/usr/bin/env python
import numpy as np
import pandas as pd

def retrieve_color(colors):
    colors = [color.split('; ')[0] for color in colors['colors'].values[0].split('; ')]

    final_colors = []
    for color in colors:
        color = color.replace('(', '').replace(')', '').split(', ')
        color = [int(value) for value in color]
        color = np.array(color).reshape(1, 1, 3)
        final_colors.append(color)

    final_colors = np.concatenate(final_colors, axis=1)

    return final_colors


def sfg(song_name):
    popular_songs = pd.read_csv("data.csv")

    song = song_name

    c = retrieve_color(popular_songs[popular_songs['track_name'] == song])
    return c
