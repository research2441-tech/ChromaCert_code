#!/bin/bash
# Download each Crello v5.1 shard, extract it, and delete the raw parquet.
cd "$(dirname "$0")"
B=https://huggingface.co/datasets/cyberagent/crello/resolve/main/data
for spec in test:4 validation:3 train:31; do
  s=${spec%%:*}; n=${spec##*:}
  for ((i=0;i<n;i++)); do
    f=$(printf "%s-%05d-of-%05d.parquet" $s $i $n)
    [ -f prep/${f%.parquet}_designs.pkl ] && continue
    curl -s -L --retry 5 -o raw/$f $B/$f && python3 prep_crello.py raw/$f $s prep && rm -f raw/$f
  done
done
echo ALLDONE
