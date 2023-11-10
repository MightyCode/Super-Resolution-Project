import os
import requests

resources_folder: str = "resources/"
def download_and_extract_dataset():
    url = 'https://drive.google.com/u/0/uc?id=1T73vDErBMBBabdc0RNCEFd28qIG1fUyY&export=download'
    save_dir = 'resources'
    os.makedirs(save_dir, exist_ok=True)

    #file_name = url.split('/')[-1]
    file_name = "pokemon.zip"
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
    bash_command = "unzip " + resources_folder + file_name + " -d " + resources_folder
    print(f'executing : {bash_command}')
    os.system(bash_command)

    os.remove(resources_folder + file_name)
    print(f'{file_name} removed after being extracted')

    female_folder = os.path.join(resources_folder, 'pokemon', 'sugimori', 'female')
    if os.path.exists(female_folder):
        bash_command = "rm -rf " + female_folder
        print(f'executing : {bash_command}')
        os.system(bash_command)

if __name__ == "__main__":
    download_and_extract_dataset()