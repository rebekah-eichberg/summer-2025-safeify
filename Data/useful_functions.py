# Code
import os
import requests
from tqdm import tqdm
import gzip
import shutil
import hashlib

def download_if_not_exists(url, filename, timeout=10):
    """
    Downloads a file from the specified URL and saves it to the given filename.

    If the file already exists, the function prints a message and does not re-download it.
    The download is streamed in chunks to support large files, and a progress bar is shown.
    A timeout can be specified to avoid hanging connections.

    Parameters:
    -----------
    url : str
        The URL to download the file from.
    filename : str
        The local file path to save the downloaded content.
    timeout : int, optional
        Timeout for the HTTP request in seconds (default is 10 seconds).

    Raises:
    -------
    requests.exceptions.Timeout
        If the request times out.
    requests.exceptions.RequestException
        For any other request-related errors.

    Returns:
    --------
    None
    """
    if os.path.exists(filename) or (filename[-3:]=='.gz' and os.path.exists(filename[:-3])):
        print("File already exists.")
        return
    

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(filename, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filename
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))

        print(f"Downloaded and saved to '{filename}'.")

    except requests.exceptions.Timeout:
        print("Download timed out.")
    except requests.RequestException as e:
        print(f"Download failed: {e}")


def extract_gz_file(gz_path, output_path=None, delete_original=False):
    """
    Extracts a .gz (Gzip) file.

    Parameters:
    -----------
    gz_path : str
        Path to the .gz file to be extracted.
    output_path : str, optional
        Path to save the extracted file. If not provided, the .gz extension is removed.
    delete_original : bool, optional
        If True, deletes the original .gz file after successful extraction.

    Returns:
    --------
    str
        Path to the extracted file.

    Raises:
    -------
    FileNotFoundError
        If the .gz file does not exist.
    OSError
        If there is an error during extraction.
    """
    if not os.path.isfile(gz_path):
        raise FileNotFoundError(f"No such file: {gz_path}")

    if output_path is None:
        output_path = gz_path[:-3] if gz_path.endswith('.gz') else gz_path + '.out'

    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted to '{output_path}'")

        if delete_original:
            os.remove(gz_path)
            print(f"Deleted original file: '{gz_path}'")

        return output_path

    except OSError as e:
        print(f"Extraction failed: {e}")
        raise



def calculate_md5(filename):
    """Calculates the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None

def verify_md5_file(md5_file_path):
    """Verifies files against MD5 checksums in a file."""
    try:
        with open(md5_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("  ", 1)
                if len(parts) != 2:
                    print(f"Invalid line format: {line}")
                    continue
                expected_md5, filename = parts
                calculated_md5 = calculate_md5(filename)

                if calculated_md5 is None:
                  print(f"File not found: {filename}")
                  continue
                if calculated_md5 == expected_md5:
                    print(f"{filename}: OK")
                else:
                    print(f"FAILED: {filename} (Expected: {expected_md5}, Calculated: {calculated_md5})")
    except FileNotFoundError:
        print(f"Error: MD5 file not found: {md5_file_path}")
