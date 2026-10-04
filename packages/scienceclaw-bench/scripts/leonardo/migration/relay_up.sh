#!/bin/bash
# Mac -> (Japan SOCKS route) -> VPS HTTP receiver. One dataset at a time: tar+gzip (nice, 1 process) -> 400 MB parts -> sha256 -> curl PUT -> delete staging.
# Light on the Mac: one gzip + one curl at a time.
SRC=/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets
ST=/private/tmp/claude-501/sc-scratch/mig/up; mkdir -p $ST
PX="--proxy socks5h://127.0.0.1:10882"; VPS=http://185.212.56.211:38422
for d in "$@"; do
  rm -rf $ST/$d; mkdir -p $ST/$d
  nice -n 15 tar -C $SRC -cf - --exclude='*.part' --exclude='download-parts' --exclude='archives' --exclude='*.zip' --exclude='*.tgz' --exclude='*.tar.gz' --exclude='*.tar.zst' --exclude='.DS_Store' $d \
    | nice -n 15 gzip -3 | split -b 400m -d -a 3 - $ST/$d/part_
  ( cd $ST/$d && shasum -a 256 part_* > SHA256SUMS )
  ok=1
  for f in $ST/$d/part_* $ST/$d/SHA256SUMS; do
    n=$(basename $f); done_one=0
    for i in 1 2 3 4 5 6; do
      code=$(curl -s -m 900 $PX -o /dev/null -w "%{http_code}" -T $f $VPS/data/$d/$n)
      [ "$code" = 201 ] && { done_one=1; break; }
      echo "retry $i $d/$n http=$code"; sleep 5
    done
    [ $done_one = 1 ] || ok=0
  done
  rm -rf $ST/$d
  [ $ok = 1 ] && echo "UP_OK $d" || echo "UP_FAIL $d"
done
echo ALL_UPLOADED
