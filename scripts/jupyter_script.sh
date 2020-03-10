#!/bin/bash

# This converts the requested .py to .ipynb
# Start the jupyter server
# Converts .ipynb back to .py once the server is closed

echo "[script] Enter number of the file you want to work on:"
files=$(ls *.py)
i=1
for j in $files
do
echo "     ($i) $j"
file[i]=$j
i=$(( i + 1 ))
done
read input
filename=${file[$input]::-3}
echo "[script] You selected the file $filename.py"

if [ -f "$filename.ipynb" ]
then
	echo "[script] $filename.ipynb already exists! Exiting.."
	exit 1
else
	echo "[script] No $filename.ipynb found. Safe to continue."
fi

echo "[script] Converting $filename.py to $filename.ipynb"
jupytext --to ipynb $filename.py

echo "[script] Trusting $filename.ipynb"
jupyter trust $filename.ipynb

echo "[script] Starting Jupyter Lab server"
jupyter lab

echo "[script] Converting $filename.ipynb back to $filename.py"
jupytext --to py $filename.ipynb

echo "[script] Removing $filename.ipynb"
rm $filename.ipynb

echo "[script] Complete"
