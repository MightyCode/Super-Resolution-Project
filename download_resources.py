import os
import requests

import gdown
import zipfile

resources_folder: str = "resources"

def download_and_extract_dataset():
    url = 'https://veekun.com/static/pokedex/downloads/pokemon-sugimori.tar.gz'
    save_dir = 'resources'
    os.makedirs(save_dir, exist_ok=True)

    file_name = url.split('/')[-1]

    file_path = os.path.join(save_dir, file_name)

    if not os.path.exists(resources_folder + file_name):
        print(f'downloading file from url : {url} ')
        response = requests.get(url)

        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded and saved to {file_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f'{resources_folder + file_name} already existes, no downloading required')

    bash_command = "tar -xvf " + resources_folder + file_name + " -C " + resources_folder
    print(f'executing : {bash_command}')
    os.system(bash_command)
    print('done')



def down_drive(url:str, dest:str):
    gdown.download(url, dest, quiet=False)

def unzip_file(file_path: str, extract_path: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def get_dataset(url:str, dest:str = "dataset"):
    folder_path = os.path.join(resources_folder, dest)
    zip_path = folder_path + ".zip"
    if os.path.exists(folder_path):
        print(f"Dataset already dowloaded at {folder_path}")
        return
    down_drive(url = url, dest = zip_path)
    print(f"Extracting to {folder_path} ...")
    unzip_file(zip_path, resources_folder)
    os.remove(zip_path)
    print("Done!")

if __name__ == "__main__":
    download_and_extract_dataset()
    get_dataset("https://drive.google.com/uc?export=download&id=16cXZ2etHDMrgXSXtLlZ5HuPh7e0KSkVk", "256")
    get_dataset("https://drive.google.com/uc?export=download&id=19WNetKoHChsVk4KDBAgnhLQEcS2ILgId", "128")