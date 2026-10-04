#!/bin/bash
# On Leonardo: pull dataset parts from the VPS receiver (HTTP), verify sha256, unpack into the data root. Idempotent; waits for parts that are not uploaded yet.
VPS=http://185.212.56.211:38422
L=/leonardo_scratch/large/userexternal/rqian000; D=$L/scienceclaw-data/datasets; T=$L/scienceclaw-data/incoming
mkdir -p $D $T
for ds in "$@"; do
  [ -f $D/.done_$ds ] && { echo "SKIP $ds"; continue; }
  mkdir -p $T/$ds; cd $T/$ds
  until curl -sf -m 60 -o SHA256SUMS $VPS/data/$ds/SHA256SUMS; do sleep 20; done      # manifest is uploaded last
  ok=1
  while read -r sum name; do
    for i in 1 2 3 4 5; do
      curl -sf -m 1200 -C - -o $name $VPS/data/$ds/$name
      [ "$(sha256sum $name | cut -d' ' -f1)" = "$sum" ] && break
      echo "retry $i $ds/$name"; rm -f $name; sleep 5
    done
    [ "$(sha256sum $name | cut -d' ' -f1)" = "$sum" ] || ok=0
  done < SHA256SUMS
  if [ $ok = 1 ]; then cat part_* | tar xzf - -C $D && touch $D/.done_$ds && echo "PULL_OK $ds $(du -sm $D/$ds | cut -f1) MB" && rm -rf $T/$ds
  else echo "PULL_FAIL $ds"; fi
done
echo ALL_PULLED
