import os
import requests

resources_folder: str = "resources/"
def download_and_extract_dataset():
    url = 'https://veekun.com/static/pokedex/downloads/pokemon-sugimori.tar.gz'
    save_dir = 'resources'
    os.makedirs(save_dir, exist_ok=True)

    file_name = url.split('/')[-1]

    file_path = os.path.join(save_dir, file_name)

    print(f'downloading file from url : {url} ')
    response = requests.get(url)

    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved to {file_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    bash_command = "tar -xvf " + resources_folder + file_name + " -C " + resources_folder
    print(f'executing : {bash_command}')
    os.system(bash_command)
    print('done')

if __name__ == "__main__":
    download_and_extract_dataset()