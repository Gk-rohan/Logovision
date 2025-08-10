import gdown
import os

# Define the Google Drive file ID and output folder
file_id = "1ksMXoPxj5QFrcsyfMKPXDtpQUxzfFVBh"
output_folder = "weights"
output_file = os.path.join(output_folder, "checkpoint_best_regular.pth")

# Create the weights folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Construct the Google Drive download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file using gdown
try:
    print(f"Downloading file from Google Drive to {output_file}...")
    gdown.download(url, output_file, quiet=False)
    print(f"File downloaded successfully to {output_file}")
except Exception as e:
    print(f"Error downloading file: {str(e)}")

# Verify the file exists
if os.path.exists(output_file):
    print(f"File is present in {output_folder}")
else:
    print(f"Failed to download file to {output_folder}")