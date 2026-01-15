"""
Download CAMELS dataset from official sources.

This script downloads the CAMELS dataset including:
- Basin attributes
- Meteorological forcing data
- Streamflow observations
- Extended forcing data (optional)
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class CAMELSDownloader:
    """Download CAMELS dataset from official sources."""

    def __init__(self, base_path: str = "./data/CAMELS_raw"):
        """
        Initialize downloader.

        Args:
            base_path: Base directory for downloaded data
        """
        self.base_path = Path(base_path)
        # self.base_url = "https://gdex.ucar.edu/dataset/camels/file/"
        self.base_url = "https://zenodo.org/records/15529996/files/"

    def download_all(self, include_extended: bool = True):
        """
        Download all CAMELS data.

        Args:
            include_extended: Whether to download extended forcing data
        """
        print("=" * 60)
        print("CAMELS Dataset Downloader")
        print("=" * 60)
        print(f"\nTarget directory: {self.base_path}")
        print(f"Include extended forcing: {include_extended}")
        print()

        # Create directories
        self._create_directories()

        # Download components
        self._download_attributes()
        self._download_forcing()

        if include_extended:
            self._download_extended_forcing()

        print("\n✓ Download complete!")
        print(f"Data saved to: {self.base_path}")

    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.base_path,
            self.base_path / "camels_attributes_v2.0",
            self.base_path / "basin_timeseries_v1p2_metForcing_obsFlow"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url: str, dest_path: Path, desc: str):
        """Download a file with progress indication."""
        if dest_path.exists():
            print(f"  {desc} already exists, skipping...")
            return

        print(f"  Downloading {desc}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"  ✓ {desc} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {desc}: {e}")
            raise

    def _download_attributes(self):
        """Download attribute files."""
        print("\n1. Downloading basin attributes...")

        attr_dir = self.base_path / "camels_attributes_v2.0"

        # Attribute files to download
        attr_files = [
            "camels_clim.txt",
            "camels_geol.txt",
            "camels_hydro.txt",
            "camels_name.txt",
            "camels_soil.txt",
            "camels_topo.txt",
            "camels_vege.txt",
            "camels_attributes_v2.0.xlsx"
        ]

        for filename in attr_files:
            url = f"{self.base_url}{filename}"
            dest = attr_dir / filename
            self._download_file(url, dest, filename)

    def _download_forcing(self):
        """Download forcing data."""
        print("\n2. Downloading forcing data...")

        forcing_dir = self.base_path / "basin_timeseries_v1p2_metForcing_obsFlow"
        zip_file = forcing_dir / "basin_timeseries_v1p2_metForcing_obsFlow.zip"

        # Download zip file
        url = f"{self.base_url}basin_timeseries_v1p2_metForcing_obsFlow.zip"
        self._download_file(url, zip_file, "Forcing data (this is large, ~3GB)")

        # Extract if needed
        if zip_file.exists() and not (forcing_dir / "basin_dataset_public_v1p2").exists():
            print("  Extracting forcing data...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(forcing_dir)
            print("  ✓ Extraction complete")

    def _download_extended_forcing(self):
        """Download extended forcing data with min/max temperature."""
        print("\n3. Downloading extended forcing data...    ")

        base_forcing = self.base_path / "basin_timeseries_v1p2_metForcing_obsFlow"
        extended_dir = base_forcing / "basin_dataset_public_v1p2" / "basin_mean_forcing"

        # Maurer extended
        print("\n  Downloading Maurer extended...")
        maurer_url = "https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/data/contents/maurer_extended.zip"
        maurer_file = extended_dir / "maurer_extended.zip"

        if not (extended_dir / "maurer_extended").exists():
            self._download_file(maurer_url, maurer_file, "Maurer extended forcing")
            if maurer_file.exists():
                print("  Extracting Maurer extended...")
                with zipfile.ZipFile(maurer_file, 'r') as zip_ref:
                    zip_ref.extractall(extended_dir)
                print("  ✓ Maurer extended ready")

        # NLDAS extended
        print("\n  Downloading NLDAS extended...")
        nldas_url = "https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/data/contents/nldas_extended.tar.xz"
        nldas_file = extended_dir / "nldas_extended.tar.xz"

        if not (extended_dir / "nldas_extended").exists():
            self._download_file(nldas_url, nldas_file, "NLDAS extended forcing")
            if nldas_file.exists():
                print("  Extracting NLDAS extended...")
                with tarfile.open(nldas_file) as tar_ref:
                    tar_ref.extractall(extended_dir)
                print("  ✓ NLDAS extended ready")


def main():
    """Main download function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download CAMELS dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/CAMELS_raw",
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--no-extended",
        action="store_true",
        help="Skip downloading extended forcing data"
    )

    args = parser.parse_args()

    # Check if data already exists
    output_path = Path(args.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        response = input(f"\nData already exists at {output_path}. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return

    # Download data
    downloader = CAMELSDownloader(args.output_dir)
    downloader.download_all(include_extended=not args.no_extended)

    print("\nNext steps:")
    print("1. Run prepare_camels.py to process the data into NetCDF format")
    print("2. Run verify_camels.py to check data integrity")


if __name__ == "__main__":
    main()