#!/bin/sh
read -p "Enter session name: " NAME
screen -dmS $NAME python -i main.py
sleep 1
screen -r $NAME