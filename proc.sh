#!/bin/bash
#
dout=data_test

#ERA5 section

lon1=35
lon2=35
lat1=54
lat2=54

ddir=/work/cmcc/lg07622/land/forcing/ERA5/spreads/TPHWL6Hrly

rm -Rf ${dout}/tmp_era.txt

for name in TBOT QBOT WIND PSRF
do
rm -Rf ${dout}/${name}_timeseriesERA.txt

while read linha; do
cdo -O remapbil,r720x360 -selvar,${name} $linha tmp/remapped_file.nc
cdo -O remapnn,lon=${lon1}_lat=${lat1} tmp/remapped_file.nc tmp/remapnn_file.nc 
cdo -outputtab,date,time,value tmp/remapnn_file.nc >> ${dout}/tmp_era.txt
echo $linha
done< <(ls ${ddir}/clmforc.0023.0.5d.TPQWL.2008-*.nc)

#remove the line "#      date     time    value" from the final dataset
grep -v 'value' ${dout}/tmp_era.txt > ${dout}/${name}_timeseriesERA.txt
rm -Rf ${dout}/tmp_era.txt


done

ddir=/work/cmcc/lg07622/land/forcing/ERA5/spreads/Precip6Hrly

for name in PRECTmms
do
rm -Rf ${dout}/${name}_timeseriesERA.txt

while read linha; do
cdo -O remapbil,r720x360 -selvar,${name} $linha tmp/remapped_file.nc
cdo -O remapnn,lon=${lon1}_lat=${lat1} tmp/remapped_file.nc tmp/remapnn_file.nc 
cdo -outputtab,date,time,value tmp/remapnn_file.nc >> ${dout}/tmp_era.txt
echo $linha
done< <(ls ${ddir}/clmforc.0023.0.5d.Prec.2008-*.nc)

#remove the line "#      date     time    value" from the final dataset
grep -v 'value' ${dout}/tmp_era.txt > ${dout}/${name}_timeseriesERA.txt
rm -Rf ${dout}/tmp_era.txt

done

ddir=/work/cmcc/lg07622/land/forcing/ERA5/spreads/Solar6Hrly

for name in FSDS
do
rm -Rf ${dout}/${name}_timeseriesERA.txt

while read linha; do
cdo -O remapbil,r720x360 -selvar,${name} $linha tmp/remapped_file.nc
cdo -O remapnn,lon=${lon1}_lat=${lat1} tmp/remapped_file.nc tmp/remapnn_file.nc 
cdo -outputtab,date,time,value tmp/remapnn_file.nc >> ${dout}/tmp_era.txt
echo $linha
done< <(ls ${ddir}/clmforc.0023.0.5d.Solr.2008-*.nc)

#remove the line "#      date     time    value" from the final dataset
grep -v 'value' ${dout}/tmp_era.txt > ${dout}/${name}_timeseriesERA.txt
rm -Rf ${dout}/tmp_era.txt
done


#CLM5 section
exp=control2
ddir=/work/cmcc/lg07622/land/work/clm5_23/${exp}/run
lon1=35
lon2=35.5
lat1=54
lat2=54.5
for name in LEAFC LEAFN H2OSNO
do
rm -Rf ${dout}/${name}_timeseries.txt
while read linha; do
cdo -outputtab,date,time,value -selvar,${name} -sellonlatbox,$lon1,$lon2,$lat1,$lat2 $linha | tail -1 >> ${dout}/${name}_timeseries.txt
echo $linha
done< <(ls ${ddir}/${exp}.clm2_0002.h0.2008-*)
done

for name in H2OSOI
do
rm -Rf ${dout}/${name}_timeseries.txt
while read linha; do
cdo -outputtab,date,time,value -selvar,${name} -sellevel,0.01 -sellonlatbox,$lon1,$lon2,$lat1,$lat2 $linha | tail -1 >> ${dout}/${name}_timeseries.txt
echo $linha
done< <(ls ${ddir}/${exp}.clm2_0002.h0.2008-*)
done



