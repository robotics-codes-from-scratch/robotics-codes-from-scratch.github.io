#!/bin/sh

# Generate RCFS logo from a linux command prompt
# Author: Sylvain Calinon, 2023

#https://fonts.google.com/specimen/Roboto (Apache 2.0 license)
#https://fonts.google.com/specimen/Permanent+Marker (Apache 2.0 license)
#Courier (public domain)
FONT1="Roboto-Black"
FONT2="Permanent-Marker-Regular"
FONT3="Courier"

convert -size 695x220 xc:white -font $FONT1 -pointsize 160 -fill 'rgb(0%,0%,0%)' -stroke 'rgb(0%,0%,0%)' -strokewidth 70 -annotate +40+168 'RCFS' -blur 0x4 -radial-blur 0x6 -ordered-dither h6x6a RCFS_bkg.png
convert -size 695x220 xc:black -font $FONT1 -pointsize 160 -fill 'rgb(100%,30%,0%)' -annotate +40+168 'RCFS' RCFS_r.png
convert -size 695x220 xc:black -font $FONT2 -pointsize 160 -fill 'rgb(0%,70%,100%)' -annotate +40+168 'RCFS' RCFS_b.png
convert -size 695x220 xc:white -font $FONT3 -pointsize 40 -fill 'rgb(0%,0%,0%)' -stroke 'rgb(0%,0%,0%)' -interline-spacing -10 \( -strokewidth 0 -annotate +495+68 'Robotics\nCodes\nFrom\nScratch.' \) \( -strokewidth 2 -annotate +495+68 'R\nC\nF\nS' \) RCFS_text.png
convert \( RCFS_r.png RCFS_b.png -compose Lighten -composite \) \( RCFS_r.png RCFS_b.png -compose Darken -composite -colorspace gray -normalize \) -compose Screen -composite RCFS_rbw.png
convert \( RCFS_bkg.png RCFS_text.png -compose Multiply -composite \) RCFS_rbw.png -compose Screen -composite -colors 16 logo-RCFS.png

