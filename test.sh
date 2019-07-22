for D in $*
do
  (cd $D ;echo $D;  awk 'BEGIN{t=0;} (!/#/ && NR%1==0){t=t+0.002*exp($NF/2.5)/1e9; print  $1,t}' BINNED_COLVAR >time.out;tail -1 time.out )
done
