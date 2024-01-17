import exiftool, os, datetime

def load_meta(fullpath: str, Height_estimator, planned_height):

    # fullpath = os.path.join(path_to_source_folder, filename)

    valuable_keys_mapping = {
        'EXIF:FocalLength': 'FocalLength',
        'EXIF:ExifImageWidth': 'ImageWidth',
        'EXIF:ExifImageHeight': 'ImageHeight',
        'EXIF:GPSLatitude': 'GPSLatitude',
        'EXIF:GPSLongitude': 'GPSLongitude',
        'XMP:AbsoluteAltitude': 'AbsoluteAltitude',
        'XMP:RelativeAltitude': 'RelativeAltitude',
        'XMP:GimbalRollDegree': 'GimbalRoll',
        'XMP:GimbalYawDegree': 'GimbalYaw',
        'XMP:GimbalPitchDegree': 'GimbalPitch',
        'XMP:FlightRollDegree': 'FlightRoll',
        'XMP:FlightYawDegree': 'FlightYaw',
        'XMP:FlightPitchDegree': 'FlightPitch',
        'EXIF:CreateDate': 'datetime'
    }

    meta = {}
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(fullpath)[0]
        for source_key, target_key in valuable_keys_mapping.items():
            val = metadata[source_key]

            if target_key == 'datetime':
                date, time = val.split()
                val = datetime.datetime(
                    *map(int, date.split(':')), 
                    *map(int, time.split(":"))
                )


            if source_key.startswith("XMP") and isinstance(val, str):
                val = float(val.strip('+'))
            meta[target_key] = val

    #################################################
        # CV2 OR PIL image loading
        # image = None

    #################################################
    # meta['HAG'] = Height_estimator.calc_hag(meta)
    meta['Planned_H'] = planned_height

    hag = None#Height_estimator.hag_interpolated(meta)
    if hag is None:
        hag = Height_estimator.calc_hag(meta)
    
    if hag is None:
        hag = planned_height

    meta['HAG'] = hag
    meta['Aprox_H'] = planned_height*0.4 + meta['HAG']*0.6

    # meta['HAG_interpolated'] = Height_estimator.hag_interpolated(meta)
    print('HAG is ', meta['HAG'], 'Aprox HAG: ', meta['Aprox_H'])

    # meta['Aprox_H'] = meta['HAG']#planned_height*0.4 + meta['HAG']*0.6
    # meta['Aprox_H_interpolated'] = planned_height*0.4 + meta['HAG_interpolated']*0.6
    # meta['Aprox_H_interpolated'] = meta['HAG_interpolated']#planned_height*0.4 + meta['HAG_interpolated']*0.6

    return meta