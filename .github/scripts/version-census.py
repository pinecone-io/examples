#! /usr/bin/env python

# This script will inspect every notebook in the repository and gather
# information about the notebooks

import os
import json
import nbformat
import re

def get_plugin_used(source):
    regex = rf"\s\"?\'?(pinecone-plugin-[a-zA-Z0-9-_]+)\"?\'?"
    match = re.search(regex, source)
    if match is None:
        return None
    return match.group(1)

def get_version(source, client):
    # escape square brackets in client name
    client = re.escape(client)
    regex = rf"\s\"?\'?{client}\"?\'?==\"?\'?([0-9]+\.[0-9]+\.[0-9]+)\"?\'?"
    match = re.search(regex, source)
    if match is None:
        return None
    return match.group(1)

def has_client(source, client):
    return f"{client}" in source

def main():
    # Track distribution of pinecone versions being used
    pinecone_versions = {}
    plugins_used = {}
    malformed_notebooks = []

    for root, _, files in os.walk(".", topdown=True):
        if '.git' in root:
            continue
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                with open(notebook_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                    for cell in nb.cells:
                        if cell.cell_type == "code":
                            if "pip" not in cell.source:
                                continue

                            plugin = get_plugin_used(cell.source)
                            if plugin is not None:
                                if plugin in plugins_used:
                                    plugins_used[plugin].append(notebook_path)
                                else:
                                    plugins_used[plugin] = [notebook_path]
                                continue

                            clients = [
                                "pinecone-client[grpc]",
                                "pinecone[grpc]",
                                "pinecone-client",
                                "langchain-pinecone",
                                "pinecone",
                            ]

                            for client in clients:
                                found_client = None
                                if has_client(cell.source, client):
                                    found_client = client
                                    break
                            if not found_client:
                                continue

                            if f"{client}==" in cell.source:
                                version = get_version(cell.source, client)
                                if version is None:
                                    print('===============================================')
                                    print(f"Could not find {client} version in {notebook_path}")
                                    print(cell.source)
                                    print('===============================================')
                                    malformed_notebooks.append(notebook_path)
                                    continue
                            else:
                                version = "unversioned"
                            
                            combined_version = f"{client}=={version}"

                            if combined_version in pinecone_versions:
                                pinecone_versions[combined_version].append(notebook_path)
                            else:
                                pinecone_versions[combined_version] = [notebook_path]

    client_types = [
        "pinecone", 
        "pinecone[grpc]",
        "pinecone-client",  
        "pinecone-client[grpc]",
        "langchain-pinecone",
    ]
    for client_type in client_types:
        print()
        print(f"Notebooks using {client_type}:")
        for version, notebooks in sorted(pinecone_versions.items()):
            client = version.split("==")[0]
            if client_type == client:
                print(f"  {version}: {len(notebooks)} notebooks")
                for notebook in notebooks:
                    print("     - ", notebook)
                print()

    print()
    print("Notebooks with malformed pinecone version specifiers:")
    for notebook in malformed_notebooks:
        print("   - ", notebook)

    print()
    print("Notebooks using plugins:")
    for plugin, notebooks in sorted(plugins_used.items()):
        print(f"  {plugin}: {len(notebooks)} notebooks")
        for notebook in notebooks:
            print("     - ", notebook)
        print()


if __name__ == "__main__":
    main()

