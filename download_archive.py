import os
import zipfile
import gdown


def download_and_extract_gdrive(file_id: str, extract_to: str = '.'):
   url = f'https://drive.google.com/uc?id={file_id}'
   temp_archive_path = 'temp_archive.zip'
   try:
        print("Starting the archive download...")
        gdown.download(url, temp_archive_path, quiet=False)
        if not os.path.exists(temp_archive_path):
            raise FileNotFoundError("Failed to download the file. Please check the file ID and sharing permissions.")

        print(f"Extracting the archive to {os.path.abspath(extract_to)}...")
        with zipfile.ZipFile(temp_archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

        print("Success! The archive has been extracted.")

   except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP archive.")
   except Exception as e:
        print(f"An error occurred: {e}")

   finally:
        if os.path.exists(temp_archive_path):
            os.remove(temp_archive_path)
            print("Temporary archive deleted.")


if __name__ == "__main__":
    folder1 = 'chroma_db'
    folder2 = 'cache'
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print(f"Required folders ('{folder1}' or '{folder2}') are missing. Starting download...")
        GOOGLE_DRIVE_FILE_ID = '1VcqrqgKzENrxDqKqvP93JOUhfCPxMRWw'
        download_and_extract_gdrive(GOOGLE_DRIVE_FILE_ID)
    else:
        print("Both folders already exist in the project root. Download skipped.")