import argparse
import os
import json
import copy

"""
Create an argument parser with a first argument for the method and n after arguments for the paths
"""
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Method used")
    parser.add_argument("arguments", nargs="+", help="Following arguments")

    return parser

RESULTS_FOLDER = "results/"

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.method == "extract":
        # Check if there is at least one path and if the path exists
        if len(args.arguments) < 2:
            print("At least 2 paths are required")
            exit()
        
        path = args.arguments[0]
        saving_path = args.arguments[1]

        if not os.path.exists(RESULTS_FOLDER + path):
            print("Path not found")
            exit()
        
        # its file
        with open(RESULTS_FOLDER + path) as f:
            data = json.load(f)
        
        if data is None:
            print("Invalid json file")
            exit()

        entries: list = data["entries"]

        save_entries = {
            "entries": []
        }

        for entry in entries:
            print(entry["model"]["name"])
            if entry["model"]["type"] == "alternative":
                save_entries["entries"].append(copy.deepcopy(entry))

        with open(RESULTS_FOLDER + saving_path, "w") as f:
            json.dump(save_entries, f, indent=4)

    elif args.method == "copy":
        # Check if there is at least one path and if the path exists
        if len(args.arguments) < 2:
            print("At least 2 paths are required")
            exit()
        
        path = args.arguments[0]
        saving_path = args.arguments[1]

        if not os.path.exists(RESULTS_FOLDER + path):
            print("Path not found")
            exit()
        
        # its file
        with open(RESULTS_FOLDER + path) as f:
            data = json.load(f)
        
        if data is None:
            print("Invalid json file")
            exit()

        with open(RESULTS_FOLDER + saving_path, "w") as f:
            json.dump(data, f, indent=4)
    
    elif args.method == "merge":
        # Check if there is at least one path and if the path exists
        if len(args.arguments) < 2:
            print("At least 2 paths are required")
            exit()
        
        path_1 = args.arguments[0]
        path_2 = args.arguments[1]

        saving_path = args.arguments[0] if len(args.arguments) < 3 else args.arguments[2]

        if not os.path.exists(RESULTS_FOLDER + path_1):
            print("Path 1 not found")
            exit()

        if not os.path.exists(RESULTS_FOLDER + path_2):
            print("Path 2 not found")
            exit()
        
        # its file
        with open(RESULTS_FOLDER + path_1) as f:
            data_1 = json.load(f)
        
        if data_1 is None:
            print("Invalid json file")
            exit()
        
        with open(RESULTS_FOLDER + path_2) as f:
            data_2 = json.load(f)
        
        if data_2 is None:
            print("Invalid json file")
            exit()

        print("Saved file at: " + RESULTS_FOLDER + saving_path)
        print("Continue? (y/n)")
        answer = input()

        if answer != "y":
            exit()
        
        for entry in data_2["entries"]:
            # Check if same entry exists so data 2 ovverrides data 1 else add

            found = False
            for i, entry_1 in enumerate(data_1["entries"]):
                if entry_1["model"]["name"] == entry["model"]["name"] \
                    and entry_1["model"]["type"] == entry["model"]["type"] \
                    and entry_1["dataset"] == entry["dataset"] \
                    and entry_1["upscaleFactor"] == entry["upscaleFactor"] \
                    and entry_1["method"]["method"] == entry["method"]["method"]:

                    found = True

                    data_1["entries"][i] = entry
                    
                    break

            if not found:
                data_1["entries"].append(entry)

        with open(RESULTS_FOLDER + saving_path, "w") as f:
            json.dump(data_1, f, indent=4)


