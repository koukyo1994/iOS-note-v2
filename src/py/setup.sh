#!/bin/sh
mkdir /usr/share/fonts/custom

for d in `ls -1 fonts`
do
  echo $d
  if [ `ls fonts/$d | grep .ttf` ]; then
    ls fonts/$d | grep .ttf | xargs -I % cp fonts/$d/% /usr/share/fonts/custom
  fi
done
