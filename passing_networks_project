#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:08:23 2022

@author: maxenceflaba
"""

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import pandas as pd 

#Open the data
parser = Sbopen()
df, related, freeze, tactics = parser.event(69301)

#Preparing the data
sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == "England Women's"].iloc[0]["index"]
mask_england = (df.type_name == 'Pass') & (df.team_name == "England Women's") & (df.index < sub) & (df.outcome_name.isnull()) & (df.sub_type_name != "Throw-in")
mask_forward = mask_england & (df.x <= df.end_x)
mask_backward = mask_england & (df.x >= df.end_x)
mask_backward_treshold = mask_england & (df.x >= df.end_x) & (df.x - df.end_x > 2)
masks = [mask_england, mask_forward, mask_backward, mask_backward_treshold]
masks_names = ['Overall Passes', 'Forward Passes', 'Backward Passes', 'At least 5m Backward Passes']

# Plotting 
#pitch = Pitch(line_color='grey')
#fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                    endnote_height=0.04, title_space=0, endnote_space=0)
#draw 2x2 pitches
pitch = Pitch(line_color='black', pad_top=20)
fig, axs = pitch.grid(ncols = 2, nrows = 2, grid_height=0.85, title_height=0.06, axis=False,
                      endnote_height=0.04, title_space=0.04, endnote_space=0.01)
 
for i,ax in zip(range(len(masks_names)),axs['pitch'].flat[:len(masks_names)]):
    ax.text(60, -10, masks_names[i],
           ha='center', va='center', fontsize=14, fontweight='bold')
    df_pass = df.loc[masks[i], ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
    #adjusting that only the surname of a player is presented.
    df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
    df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])

    # Calculating vertices size and location
    scatter_df = pd.DataFrame()
    for i, name in enumerate(df_pass["player_name"].unique()):
        passx = df_pass.loc[df_pass["player_name"] == name]["x"].to_numpy()
        recx = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
        passy = df_pass.loc[df_pass["player_name"] == name]["y"].to_numpy()
        recy = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "player_name"] = name
        #make sure that x and y location for each circle representing the player is the average of passes and receptions
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        #calculate number of passes
        scatter_df.at[i, "no"] = df_pass.loc[df_pass["player_name"] == name].count().iloc[0]

    #adjust the size of a circle so that the player who made more passes 
    scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 700)

    # Calculating edges width
    df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
    lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
    lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
    #setting a treshold. You can try to investigate how it changes when you change it.
    lines_df = lines_df[lines_df['pass_count']>0]
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax, zorder = 3)
    for i, row in scatter_df.iterrows():
         pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=10, ax=ax, zorder = 4)
    
    for i, row in lines_df.iterrows():
        player1 = row["pair_key"].split("_")[0]
        player2 = row['pair_key'].split("_")[1]
        #take the average location of players to plot a line between them 
        player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
        player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
        player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
        player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
        num_passes = row["pass_count"]
        #adjust the line width so that the more passes, the wider the line
        line_width = (num_passes / lines_df['pass_count'].max() * 7)
        #plot lines on the pitch
        pitch.lines(player1_x, player1_y, player2_x, player2_y,
                        alpha=1, lw=line_width, zorder=2, color="red", ax = ax)
    #Centralisation
    no_passes = df_pass.groupby(['player_name']).x.count().reset_index()
    no_passes.rename({'x':'pass_count'}, axis='columns', inplace=True)
    #find one who made most passes
    max_no = no_passes["pass_count"].max() 
    #calculate the denominator - 10*the total sum of passes
    denominator = 10*no_passes["pass_count"].sum() 
    #calculate the nominator
    nominator = (max_no - no_passes["pass_count"]).sum()
    #calculate the centralisation index
    centralisation_index = nominator/denominator    
    s = "Centralisation : C = " + str(round(centralisation_index*100,2)) + " %"
    ax.text(60, 87, s,
       ha='center', va='center', fontsize=12, color='white', backgroundcolor = 'red')


axs['title'].text(0.5, 0.5, 'England passing network against Sweden', ha='center', va='center', fontsize=30, fontweight='heavy')
resolution_value = 600
plt.savefig("myImage.png", format="png", dpi=resolution_value)
plt.show()
