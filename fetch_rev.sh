#!/bin/bash
cd "$(dirname "$0")"
B=https://huggingface.co/datasets/cyberagent/crello/resolve/main/data
for ((i=30;i>=0;i--)); do
  f=$(printf "train-%05d-of-00031.parquet" $i)
  [ -f prep/${f%.parquet}_designs.pkl ] && continue
  [ -f raw/$f ] && continue
  curl -s -L --retry 5 -o raw/r_$f $B/$f && mv raw/r_$f raw/rev_$f && python3 prep_crello.py raw/rev_$f train prep && rm -f raw/rev_$f
  [ -f prep/rev_${f%.parquet}_designs.pkl ] && for k in designs crops; do mv prep/rev_${f%.parquet}_$k.pkl prep/${f%.parquet}_$k.pkl; done
done
echo REVDONE
