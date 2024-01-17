import numpy as np
from osgeo import osr, gdal
import rasterio

def calc_coords(
        meta: dict, 
        targets: np.ndarray,
        height_estimator
    ):

    # targets supposed to be in format [xmin, ymin, xmax, ymax]

    # targets = np.asfarray(targets)

    targets_centers = calc_centers(targets.cpu().numpy())
    closure = np.sum((targets_centers-0.5)**2, axis=1)

    local_displacement = pix_to_meters(targets_centers, meta)
    global_displacement = rotate_to_north(local_displacement, meta)
    targets_coords = meters_to_degrees(global_displacement, meta)
    return targets_coords, closure


def pix_to_meters_with_heights(targets_centers, targets_heights, meta):
    image_center = np.array([
        meta['ImageWidth'], meta['ImageHeight'] 
    ], dtype=np.float64)

    scale = np.array([
        3.3e-6*targets_heights/(meta['FocalLength']*1e-3) # phys_dist/height = pixel_dist/focal_le
    ], dtype=np.float64)

    displacement = ((targets_centers - 0.5)*image_center)*scale.T # maybe Y should be reversed because top left is 0,0
    displacement[:, 1] *=-1

    return displacement


def calc_centers(
        targets: np.ndarray
    ):
    targets= targets.astype(np.float64)

    targets[:, 0] += targets[:,2]/2
    targets[:, 1] += targets[:,3]/2

    return targets[:, :2]


def pix_to_meters(targets_centers, meta):

    # pixel size 3.3 μm

    # focal length 12.3mm - real 12.29 https://sdk-forum.dji.net/hc/en-us/articles/12325496609689-What-is-the-custom-camera-parameters-for-Mavic-3-Enterprise-series-and-Mavic-3M-

    image_center = np.array([
        meta['ImageWidth'], meta['ImageHeight'] 
    ], dtype=np.float64)

    scale = np.array([
        3.3e-6*meta['Aprox_H']/(meta['FocalLength']*1e-3) # phys_dist/height = pixel_dist/focal_le
    ], dtype=np.float64)

    displacement = ((targets_centers - 0.5)*image_center)*scale # maybe Y should be reversed because top left is 0,0
    displacement[:, 1] *=-1

    return displacement


def rotate_to_north(loc_displ, meta):

    angle = -meta['GimbalYaw']*np.pi/180

    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],# x*cos(a) - y*sin(a)
        [np.sin(angle), np.cos(angle)] # x*sin(a) + y*cos(a)
    ], dtype=np.float64)

    return np.matmul(rot_matrix, loc_displ.T, dtype=np.float64).T


def meters_to_degrees(displ, meta):

    radius = 6378137.0
    radius_deg = radius*np.pi/180

    scale = np.array([ 
        (
        radius_deg*np.cos(meta['GPSLatitude']*np.pi/180)
        ), 
        radius_deg
    ], dtype=np.float64)

    ref_point = np.array([
        meta['GPSLongitude'], meta['GPSLatitude']
    ], dtype=np.float64)

    return displ/scale + ref_point



class Map:
    def __init__(self, path_to_map):
        self.path = path_to_map
        try:
            ds = gdal.Open(path_to_map)
            # self.height_map = gdal.OpenEx(self.path)
            self.height_map = ds.GetRasterBand(1).ReadAsArray()
            
            # with rasterio.open('/home/user/dsm/230818100456_AsterGdemV3_6.tif') as dataset:
            #     self.height_map=dataset.read()[0]
            self.trans_loc_glob = self.create_transform(ds)
            self.height = ds.RasterYSize
            self.width = ds.RasterXSize
            self.PixtoGeoTransform = ds.GetGeoTransform() 
        
            self.trans_glob_loc = self.trans_loc_glob.GetInverse()
            self.GeotoPixTransform = gdal.InvGeoTransform(self.PixtoGeoTransform)
            self.active = True
        except Exception as e:
            print(f'Map Unavaliable: {path_to_map}, ', e)
            self.active = False


    @staticmethod
    def create_transform(cooord_map):
        old_cs= osr.SpatialReference()
        old_cs.ImportFromWkt(
            cooord_map.GetProjectionRef())
        
        wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
        new_cs = osr.SpatialReference()
        new_cs .ImportFromWkt(wgs84_wkt)

        return osr.CoordinateTransformation(old_cs, new_cs) 
    

    def to_coord(self, pix_coord):
        return gdal.ApplyGeoTransform(
            self.PixtoGeoTransform, 
            pix_coord
        )

    
    def to_pixel(self, coord):
        geo_coord, _ = self.trans_glob_loc.TransformPoint(coord)
        return [
            int(x+0.5) for x in 
            gdal.ApplyGeoTransform(
                self.GeotoPixTransform, 
                geo_coord
            )
        ]

    
    def is_point_on_map(self, coord, return_points=False):
        y, x = self.to_pixel(coord)

        onmap = (
            (x < self.width and x>=0)
            and
            (y < self.height and y>=0))

        if return_points:
            return onmap, x, y
        else:
            return onmap
        
    
    def get_pix_height(self, pix_coord):
        # ds = gdal.OpenEx(self.path)
        # return self.height_map.ReadAsArray(x, y, 1, 1)
        return self.height_map[pix_coord]

    def get_height(self, coord):
        onmap, x, y = self.is_point_on_map(
            coord, return_points=True
        )
        height = None
        if onmap:
            height = self.get_pix_height(x, y)
        return height 
    



class Height_Estimator:
    def __init__(
            self, 
            maps_paths,
    ):
        self.maps=[]
        for path in maps_paths:
            demmap = Map(path)
            if demmap.active:
                self.maps.append(demmap)

        print(f'Total Maps Number: {len(self.maps)}')
        # self.maps = [
        #     Map(path) for path in maps_paths
        # ]

    
    def interpolate_idw(self, coord):

        cur_dem = None
        # px, py = None, None

        for dem in self.maps:
            onmap, x, y = dem.is_point_on_map(
                coord, return_points=True)
            if onmap:
                cur_dem = dem
                break
                

        if cur_dem is None:
            return None

        pixel = np.array([x, y])

        shift = [-1, 0, 1]

        x_shift,y_shift = np.meshgrid(
            shift, shift)
        
        sh = np.stack(
            (x_shift,y_shift), 
            axis=2
        ).reshape(-1, 2)

        surround_pix = pixel+sh

        outlier_mask = np.logical_or(
            surround_pix<0, surround_pix>=[cur_dem.width, cur_dem.height]
        ).sum(axis=1)>0
        # mask= surround_pix<0
        # outlier_mask = np.logical_or(
        #     mask[:, 0], mask[:, 1])
        
        outlier_pix = surround_pix[
            outlier_mask]

        surround_pix = surround_pix[
            np.logical_not(outlier_mask)]
        
        surround_coords = np.array([
            cur_dem.to_coord(*coord) 
            for coord in surround_pix
        ])

        distances = self.haversine(
            np.array(coord),
            surround_coords
            )

        heights = np.array([
            cur_dem.get_pix_height(*pix)[0][0]
            for pix in surround_pix
        ]) 

        if len(outlier_pix):

            outlier_coords = np.array([
                cur_dem.to_coord(*coord) 
                for coord in outlier_pix
            ])

            outliers_heights, not_none_mask = self.height_from_coord_(
                outlier_coords
            )

            outliers_heights = outliers_heights[
                not_none_mask
            ]

            outlier_coords = outlier_coords[
                not_none_mask
            ]

            outlier_dists = self.haversine(
                np.array(coord),
                outlier_coords
            )

            distances = np.concatenate(
                (distances, outlier_dists)
            )

            heights = np.concatenate(
                (heights, outliers_heights)
            )
        

        if len(distances) and len(heights) and len(distances) == len(heights):
            return np.sum(
                heights.reshape(-1)/distances
            )/np.sum(1/distances)
        
        # return height


    def haversine(self, p1, p2):

        if len(p2)==0:
            return p2

        R = 6378137

        delta_lat = (p1[1] - p2[:, 1])*np.pi/360 
        delta_lon = (p1[0] - p2[:, 0])*np.pi/360

        a = np.sin(delta_lat)**2 + np.cos(p1[1])*np.cos(p2[:, 1])*np.sin(delta_lon)**2
        c = 2*np.arcsin(a**0.5)
        return c*R
    

    def height_from_coord_(self, coordinates):

        heights = []
        not_none = []

        for coord in coordinates:
            height = None
            for dem in self.maps:
                h = dem.get_height()
                if h is not None:
                    height = h[0][0]
            heights.append(height)
            not_none.append(height is not None)

        return np.array(heights), np.array(not_none)


    def height_from_coord_interpolated(self, coordinates):
        #coordinate assumed to be lon,lat

        heights = []
        not_none = []

        for coord in coordinates:
            height = None
            # for dem in self.maps:
            height = self.interpolate_idw()

            heights.append(height)
            not_none.append(height is not None)

        return np.array(heights), np.array(not_none)


    def height(self, coord):
        for dem in self.maps:
            h = dem.get_height(coord)
            if h is not None:
                return h[0][0]
            

    def calc_hag(self, meta):
        abs_h = meta['AbsoluteAltitude']
        ground_h = self.height(
            meta['GPSLongitude'], meta['GPSLatitude']
        )
        if ground_h is not None:
            return abs_h - ground_h
    

    def hag_interpolated(self, meta):
        abs_h = meta['AbsoluteAltitude']
        ground_h = self.interpolate_idw(
            meta['GPSLongitude'], meta['GPSLatitude']
        ) 
        if ground_h is not None:
            return abs_h - ground_h
