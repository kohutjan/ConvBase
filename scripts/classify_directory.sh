#!/bin/sh

PATH_FILE="path_file.txt"
rm $PATH_FILE

for file in "$2"/*
do
  echo "$file" >> "$PATH_FILE"
done

./../build/classify_images "$1" "$PATH_FILE"




