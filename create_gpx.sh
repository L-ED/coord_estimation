
dir=/run/user/1000/gvfs/mtp\:host\=DJI_DJI_RC_Pro_Enterprise_5YSZKB10020SML/Внутренний\ общий\ накопитель/DJI/Mission/KML/

# cd "${dir}"
# newest_flight_dir=$(ls -td -- */ | egrep -v '__pycache__' | head -n 1)
# # echo  $newest_flight_dir
# cd /home/user/Emergency_Search/path_plans

# full_path="${dir}/${newest_flight_dir}"
# filename=$(find "${full_path}" -maxdepth 1 -name "*.kmz")
# unzip "${filename}"

# num_of_flights=$(find . -name "path*" | wc -l)
# echo $num_of_flights
# f=$(expr $num_of_flights + 1)

# gpsbabel -i kml -f ./wpmz/template.kml -o gpx -F ./path$f.gpx
# rm -r ./wpmz


dest=/home/user/Emergency_Search/results

ls "${dest}" -td

newest_flight_dir=$(ls "${dest}" -t | grep result | head -n 1)
# echo $newest_flight_dir

logdir=/run/user/1000/gvfs/mtp:host=DJI_DJI_RC_Pro_Enterprise_5YSZKB10020SML/Внутренний\ общий\ накопитель/DJI/com.dji.industry.pilot/FlightRecord
file=$(ls "${logdir}" -t | grep txt | head -n 1)

cp "${logdir}/${file}" "${dest}/${newest_flight_dir}"