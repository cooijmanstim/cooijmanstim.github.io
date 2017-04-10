# something like
for dir in ~/results/mscoconet/sample_firstemd_*_2017-04-09*; do rsync  -av --prune-empty-dirs --include '*/' --include '*.gif' --exclude '*' $dir .; done

