import os
import sys
import shutil
import zipfile
import gdown


def download_and_extract_gdrive(file_id: str, extract_to: str = '.', folders_to_clean: list = None):
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

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An error occurred during download or extraction: {e}")

        if folders_to_clean:
            print("Starting cleanup of incomplete or corrupted data...")
            for folder in folders_to_clean:
                folder_path = os.path.join(extract_to, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path, ignore_errors=True)
                    print(f"Deleted corrupted directory: {folder_path}")
        raise

    finally:
        if os.path.exists(temp_archive_path):
            os.remove(temp_archive_path)
            print("Temporary ZIP archive deleted.")


if __name__ == "__main__":
    folder1 = 'chroma_db'
    folder2 = 'cache'

    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print(f"Required folders ('{folder1}' or '{folder2}') are missing. Starting download...")
        GOOGLE_DRIVE_FILE_ID = '1VcqrqgKzENrxDqKqvP93JOUhfCPxMRWw'

        try:
            download_and_extract_gdrive(
                file_id=GOOGLE_DRIVE_FILE_ID,
                extract_to='.',
                folders_to_clean=[folder1, folder2]
            )
        except Exception:

            print("Further execution is impossible. Terminating the script.")
            sys.exit(1)
    else:
        print("Both folders already exist in the project root. Download skipped.")