import json
import os
with open('indexdata.json') as json_file:
    json_data = json.load(json_file)
def write_json(data, filename='indexdata.json'):
    with open(filename,'w') as f:
        json.dump(data, f, indent=4)
def add_user(name,section,ids):
	for sect in json_data:
		json_data[sect]["tag"].append(ids)
		json_data[sect]["name"].append(name)
		print(json_data[sect])
	write_json(json_data)

