#!/bin/bash

content=$(<file.txt)

# the hf token is in "..."
extracted=$(echo "$content" | grep -oP '"\K[^"]*')

# use the std output as return
echo "$extracted"