import synapseclient
import synapseutils
import os

def download_brats_data():
    # 1. Initialize the client
    syn = synapseclient.Synapse(configPath=".synapseConfig")

    # 2. Log in using the .synapseConfig file
    syn.login()

    # 3. Define the synID for the BraTS GLI dataset FOLDER
    gli_folder_id = "syn59059776"

    # 4. Define our target directory based on our project blueprint
    target_dir = "./data/"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Initiating bulk download for folder {gli_folder_id}...")
    print("This will take some time depending on your internet connection.")
    
    # 5. Execute the recursive folder download
    files = synapseutils.syncFromSynapse(syn, gli_folder_id, path=target_dir)

    print(f"Download complete! Successfully pulled {len(files)} files to {target_dir}")

if __name__ == "__main__":
    download_brats_data()