# something like
for dir in /data/lisatmp4/cooijmat/run/mscoconet/sample_emd2_deepish_contiguish*; do rsync  -av --prune-empty-dirs --include '*/' --include '*.gif' --exclude '*' $dir .; done

