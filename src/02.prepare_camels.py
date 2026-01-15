#!/usr/bin/env python3
"""
Prepare CAMELS dataset for RainFlow.

This script processes raw CAMELS data into an efficient NetCDF format
that can be easily loaded by RainFlow's data loaders.

Note:
daymet: in the raw data sometimes tmin>tmax
areas are different between static attributes and forcing files
"""

import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.append(str(Path(__file__).parent.parent.parent))

#from rainflow.data.pet_processor import PETProcessor


class CAMELSProcessor:
    """Process CAMELS data into NetCDF format."""

    def __init__(self, raw_data_path: str, output_path: str):
        """
        Initialize processor.

        Args:
            raw_data_path: Path to raw CAMELS data
            output_path: Path to save processed data
        """
        self.raw_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize PET processor
        #self.pet_processor = PETProcessor(method='hargreaves')

        # Define forcing variables (including PET)
        self.forcing_vars = {
            'daymet': ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'dayl', 'swe', ],
            'nldas_extended': ['PRCP', 'SRAD', 'Tmax', 'Tmin', 'Vp', 'Dayl', 'SWE', 'PET'],
            'maurer_extended': ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'dayl', 'swe', 'pet']
        }
        self.var_names = ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'dayl', 'swe']  # consistent naming

        # Define static attribute variables
        self.static_vars = ['p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq',
                            'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2',
                            'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac',
                            'dom_land_cover', 'root_depth_50', 'soil_depth_pelletier', 'soil_depth_statsgo',
                            'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac',
                            'clay_frac', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
                            'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

    def process(
            self,
            start_date: str = "1980-01-01",
            end_date: str = "2014-12-31",
            basin_list_file: Optional[str] = None
    ) -> str:
        """
        Process CAMELS data to NetCDF.

        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            basin_list_file: Optional file with list of basins to process

        Returns:
            Path to output NetCDF file
        """
        print("=" * 60)
        print("CAMELS Data Processor with PET Calculation")
        print("=" * 60)
        print(f"\nProcessing data from: {self.raw_path}")
        print(f"Date range: {start_date} to {end_date}")

        # Get list of basins to process
        if basin_list_file and Path(basin_list_file).exists():
            basins = self._load_basin_list(basin_list_file)
            print(f"Processing {len(basins)} basins from list")
        else:
            basins = self._get_all_basins()
            print(f"Processing all {len(basins)} basins")

        # Load basin attributes
        print("\n1. Loading basin attributes...")
        attributes_df, lat, lon = self._load_attributes(basins)

        # Load forcing and calculate PET
        print("\n2. Loading forcing data and calculating PET...")
        forcing_data = self._load_forcing_with_pet(basins, lat, start_date, end_date)

        # Load streamflow
        print("\n3. Loading streamflow data...")
        streamflow_data = self._load_streamflow(basins, start_date, end_date)

        # Create NetCDF
        print("\n4. Creating NetCDF dataset...")
        output_file = self._create_netcdf(
            basins, lat, lon,
            forcing_data, streamflow_data, attributes_df,
            start_date, end_date
        )

        print(f"\n✓ Processing complete!")
        print(f"Output file: {output_file}")

        return output_file

    def _load_basin_list(self, filepath: str) -> List[str]:
        """Load basin list from a text file."""
        with open(filepath, 'r') as f:
            basins = [line.strip() for line in f.readlines()]
        return basins

    def _get_all_basins(self) -> List[str]:
        """Get all available basin IDs from gauge information."""
        gauge_file = self.raw_path / "basin_timeseries_v1p2_metForcing_obsFlow" / \
                     "basin_dataset_public_v1p2" / "basin_metadata" / "gauge_information.txt"

        if not gauge_file.exists():
            raise FileNotFoundError(f"Gauge information file not found: {gauge_file}")

        df = pd.read_csv(
            gauge_file,
            sep=r'\t+',
            engine='python',
            header=None,
            skiprows=1,
            names=['HUC_02', 'GAGE_ID', 'GAGE_NAME', 'LAT', 'LONG', 'DRAINAGE_AREA_KM2'],
            dtype=str
        )
        df['GAGE_ID'] = df['GAGE_ID'].str.strip()
        df = df.rename(columns={'GAGE_ID': 'gauge_id'})
        return df['gauge_id'].tolist()

    def _load_attributes(self, basins: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load static basin attributes and extract coordinates."""
        attr_path = self.raw_path / "camels_attributes_v2.0"

        # Read all attribute files
        dfs = []
        for attr_file in attr_path.glob("camels_*.txt"):
            if attr_file.stem == "camels_name":
                continue
            df = pd.read_csv(attr_file, sep=';', dtype={'gauge_id': str})
            df = df.set_index('gauge_id')
            dfs.append(df)

        # Combine all attributes
        all_attrs = pd.concat(dfs, axis=1)

        # Subset to requested basins
        missing = set(basins) - set(all_attrs.index)
        if missing:
            raise KeyError(f"Static attributes not found for basins: {sorted(missing)}")
        all_attrs = all_attrs.loc[basins]

        # Extract latitude and longitude
        lat = all_attrs['gauge_lat'].values
        lon = all_attrs['gauge_lon'].values

        # Select available static variables
        # available_vars = [v for v in self.static_vars if v in all_attrs.columns]
        attrs_df = all_attrs[self.static_vars]

        return attrs_df, lat, lon

    def _load_forcing_with_pet(
            self,
            basins: List[str],
            lat: np.ndarray,
            start_date: str,
            end_date: str
    ) -> Dict[str, np.ndarray]:
        """
        Load forcing data for all datasets and calculate PET.
        Uses a progress bar for basin loading and caches file lists to speed up globbing.
        """
        date_range = pd.date_range(start_date, end_date)
        n_days = len(date_range)
        n_basins = len(basins)

        forcing_data: Dict[str, np.ndarray] = {}

        for dataset in ['daymet', 'nldas_extended', 'maurer_extended']:
            print(f"  Loading {dataset}...")

            # Path to dataset forcing directory
            dataset_path = self.raw_path / "basin_timeseries_v1p2_metForcing_obsFlow" / \
                           "basin_dataset_public_v1p2" / "basin_mean_forcing" / dataset
            if not dataset_path.exists():
                print(f"    Warning: {dataset} not found, skipping...")
                continue

            # Pre-cache leap-year forcing files
            leap_files = list(dataset_path.glob("**/*_forcing_leap.txt"))
            # If no leap-year files found, prepare non-leap-year fallback
            nonleap_files = list(dataset_path.glob("**/*_forcing.txt")) if not leap_files else []

            # Initialize data array: (n_basins, n_days, n_vars)
            n_vars = len(self.forcing_vars[dataset])
            data = np.full((n_basins, n_days, n_vars), np.nan)

            # Iterate over basins with progress bar
            for i, basin in enumerate(tqdm(basins, desc=f"    {dataset} basins")):
                # Match leap files first, then non-leap
                candidates = [f for f in leap_files if f.name.startswith(basin)]
                if not candidates and nonleap_files:
                    candidates = [f for f in nonleap_files if f.name.startswith(basin)]
                basin_file = candidates[0] if candidates else None

                if basin_file:
                    df = self._read_forcing_file(basin_file)
                    df = df.reindex(date_range)

                    # Extract original forcing variables (exclude PET)
                    original_vars = self.forcing_vars[dataset][:-1]
                    for j, var in enumerate(original_vars):
                        if var in df.columns:
                            data[i, :, j] = df[var].values.astype(float)

                    # Determine indices for tmax and tmin
                    tmax_idx, tmin_idx = 2, 3

                    # Retrieve temperature series
                    tmax_data = data[i, :, tmax_idx]
                    tmin_data = data[i, :, tmin_idx]

                    # If tmin > tmax, swap their values
                    mask = tmin_data > tmax_data
                    if mask.any():
                        tmax_orig = tmax_data[mask].copy()
                        tmin_orig = tmin_data[mask].copy()
                        tmax_data[mask] = tmin_orig
                        tmin_data[mask] = tmax_orig

                    # Calculate PET if temperature data is available
                    # if not np.all(np.isnan(tmax_data)) and not np.all(np.isnan(tmin_data)):
                    #     pet_values = self.pet_processor._calculate_hargreaves(
                    #         tmax=tmax_data,
                    #         tmin=tmin_data,
                    #         lat=lat[i],
                    #         dates=date_range
                    #     )
                    #     data[i, :, -1] = pet_values
                else:
                    raise FileNotFoundError(
                        f"Forcing file not found for basin {basin} in {dataset_path}"
                    )

            # Store unified variable arrays in the output dict
            prefix = dataset.replace('_extended', '')
            for j, var in enumerate(self.var_names):
                var_name = f"{prefix}_{var}"
                forcing_data[var_name] = data[:, :, j]
                print(f"    Added variable: {var_name}")

            # calculate tmean
            tmax_arr = forcing_data[f"{prefix}_tmax"]  # shape (n_basins, n_days)
            tmin_arr = forcing_data[f"{prefix}_tmin"]
            tmean_arr = 0.5 * (tmax_arr + tmin_arr)
            forcing_data[f"{prefix}_tmean"] = tmean_arr
            print(f"    Added variable: {prefix}_tmean")

        return forcing_data

    def _load_streamflow(self, basins: List[str], start_date: str, end_date: str) -> np.ndarray:
        """
        Load streamflow data, with robust file matching:
        1) cache all '*streamflow*.txt' files
        2) match basin prefix against QC and non-QC filenames
        3) warn & fill NaN when missing
        """
        date_range = pd.date_range(start_date, end_date)
        n_days = len(date_range)
        n_basins = len(basins)
        flow_data = np.full((n_basins, n_days), np.nan, dtype=np.float32)

        flow_path = (self.raw_path /
                     "basin_timeseries_v1p2_metForcing_obsFlow" /
                     "basin_dataset_public_v1p2" /
                     "usgs_streamflow")

        # 1) Pre-cache all streamflow files (QC & non-QC)
        all_files = list(flow_path.glob("**/*streamflow*.txt"))

        for i, basin in enumerate(tqdm(basins)):
            # 2) Try QC first, then fallback
            qc_candidates = [f for f in all_files if f.name.startswith(f"{basin}_") and "_qc" in f.name]
            # noqc_candidates = [f for f in all_files if f.name.startswith(f"{basin}_") and "_qc" not in f.name]

            if qc_candidates:
                flow_file = qc_candidates[0]
            # elif noqc_candidates:
            #     flow_file = noqc_candidates[0]
            else:
                raise FileNotFoundError(f"No streamflow file found for basin {basin} in {flow_path}")

            df = pd.read_csv(
                flow_file, sep=r'\s+', header=None,
                names=['basin', 'year', 'month', 'day', 'flow', 'flag']
            )
            # build date index
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df = df.set_index('date').reindex(date_range)

            # Optional: drop flagged records
            # df.loc[df['flag'] != 0, 'flow'] = np.nan

            flow_data[i, :] = df['flow'].values.astype(np.float32)

        # set negative values to NaN
        flow_data[flow_data < 0] = np.nan

        return flow_data

    def _find_basin_forcing_file(self, dataset_path: Path, basin: str) -> Optional[Path]:
        """
        Search for the forcing file associated with a specific basin.

        Parameters:
            dataset_path (Path): Root directory containing all basin forcing files.
            basin (str): Basin identifier (e.g., "01013500").

        Returns:
            Optional[Path]: Path to the first matching forcing file, or None if no file is found.
        """
        files = list(dataset_path.glob("**/*_forcing_leap.txt"))
        if not files:
            files = list(dataset_path.glob("**/*_forcing.txt"))
        for f in files:
            if f.name.startswith(basin):
                return f
        return None

    def _read_forcing_file(self, filepath: Path) -> pd.DataFrame:
        """Read a forcing file and return a DataFrame indexed by date."""
        df = pd.read_csv(filepath, sep=r'\s+', header=3, dtype=str)
        df['date'] = pd.to_datetime(
            df[['Year', 'Mnth', 'Day']].astype(str).agg('-'.join, axis=1)
        )
        df = df.set_index('date')
        df.columns = [col.split('(')[0] for col in df.columns]
        return df

    def _get_area_from_header(self, basin: str, dataset: str) -> Optional[int]:
        """
        Try to read watershed area (km²) from the 3rd line of the forcing file.
        It first looks for *_forcing_leap.txt; if none, falls back to *_forcing.txt.
        Returns the integer area if found, or None.
        """
        base_dir = (
                self.raw_path
                / "basin_timeseries_v1p2_metForcing_obsFlow"
                / "basin_dataset_public_v1p2"
                / "basin_mean_forcing"
                / dataset
        )
        # collect all leap-year files under this dataset
        leap_files = list(base_dir.glob("**/*_forcing_leap.txt"))
        # if no leap files, collect non-leap
        nonleap_files = [] if leap_files else list(base_dir.glob("**/*_forcing.txt"))

        # narrow down to files that start with our basin ID
        candidates = [f for f in leap_files if f.name.startswith(basin)]
        if not candidates:
            candidates = [f for f in nonleap_files if f.name.startswith(basin)]
        if not candidates:
            return None

        # read the first match
        filepath = candidates[0]
        with open(filepath, 'r') as fp:
            header = fp.readlines()
        return int(header[2].strip())

    def _get_area(self, basin: str) -> float:
        """
        Loop through preferred datasets in order and return the first valid area.
        Fallback to the static attribute if all forcing datasets are missing.
        """
        for ds in ['daymet', 'maurer_extended', 'nldas_extended']:
            area = self._get_area_from_header(basin, ds)
            if area is not None:
                return float(area)

        # final fallback: use CAMELS static attribute
        return float(self.attributes_df.loc[basin, 'area_gages2'])

    def _create_netcdf(
            self,
            basins: List[str],
            lat: np.ndarray,
            lon: np.ndarray,
            forcing_data: Dict[str, np.ndarray],
            streamflow_data: np.ndarray,
            attributes_df: pd.DataFrame,
            start_date: str,
            end_date: str
    ) -> str:
        """Create NetCDF dataset from processed arrays."""
        time_range = pd.date_range(start_date, end_date)
        ds = xr.Dataset(
            coords={
                'station_ids': basins,
                'time': time_range,
                'lat': ('station_ids', lat),
                'lon': ('station_ids', lon)
            }
        )

        # Add forcing variables
        for var_name, data in forcing_data.items():
            ds[var_name] = (['station_ids', 'time'], data)

        # Add variable attributes
        for var_name in ds.data_vars:
            if var_name.endswith('_tmean'):
                ds[var_name].attrs['long_name'] = 'Mean daily temperature'
                ds[var_name].attrs['units'] = 'degC'

        # Add observed streamflow
        ds['QObs'] = (['station_ids', 'time'], streamflow_data)
        ds['QObs'].attrs['units'] = 'ft3 s-1'
        ds['QObs'].attrs['long_name'] = 'Observed streamflow'

        # Compute runoff [mm/day] and add to dataset, conversion factor: 2.448 = 0.0283168 * 86400 * 1000
        # area_km2 = attributes_df['area_gages2'].values.astype(float)  # shape (n_basins,)

        area_list = [ self._get_area(basin) for basin in basins ]
        area_km2 = np.array(area_list, dtype=float)

        area_km2_expanded = area_km2[:, np.newaxis]
        runoff_mm = (
                streamflow_data
                * 0.028316846592  # ft3 → m3
                * 86400  # s → day
                # / (area_km2_expanded * 1e6)  # m3/day → m/day
                / (area_km2_expanded)  # m3/day → m/day
                * 1000  # m → mm
        )
        ds['Runoff'] = (['station_ids', 'time'], runoff_mm)
        ds['Runoff'].attrs['units'] = 'mm day-1'
        ds['Runoff'].attrs['long_name'] = 'Runoff'

        # Add static attributes
        for var in attributes_df.columns:
            ds[f'{var}'] = (['station_ids'], attributes_df[var].values)

        # Add global metadata
        ds.attrs['title'] = 'CAMELS Dataset for RainFlow with PET'
        ds.attrs['source'] = 'https://ral.ucar.edu/solutions/products/camels'
        ds.attrs['processing_date'] = pd.Timestamp.now().isoformat()
        ds.attrs['date_range'] = f"{start_date} to {end_date}"
        ds.attrs['pet_method'] = 'Hargreaves'
        ds.attrs['variable_naming'] = \
            'Unified naming: {dataset}_{variable} for forcing, static_{variable} for attributes'

        # Save to NetCDF
        output_file = self.output_path / "CAMELS.nc"
        ds.to_netcdf(output_file, format='NETCDF4')

        # Print summary
        print(f"\nDataset summary:")
        print(f"  - Number of basins: {len(basins)}")
        print(f"  - Time period: {start_date} to {end_date} ({len(time_range)} days)")
        print(f"  - Forcing variables: {len(forcing_data)}")
        print(f"  - Static variables: {len(attributes_df.columns)}")

        return str(output_file)


def main():
    """Main processing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process CAMELS data to NetCDF with PET")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/CAMELS_raw",
        help="Directory with raw CAMELS data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/CAMELS_processed",
        help="Directory for processed data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1980-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2014-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--basin-list",
        type=str,
        default=None,
        help="File with list of basins to process"
    )

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        print("Please run download_camels.py first.")
        return

    processor = CAMELSProcessor(args.input_dir, args.output_dir)
    output_file = processor.process(
        start_date=args.start_date,
        end_date=args.end_date,
        basin_list_file=args.basin_list
    )

    print("\nNext steps:")
    print("Use the data in your experiments.")


if __name__ == "__main__":
    main()
