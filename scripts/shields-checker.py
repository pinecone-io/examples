import click
import logging
from pathlib import Path
import re

colab_link = re.compile(r"(?<=\[!\[Open In Colab\]\(https:\/\/colab\.research\.google\.com\/assets\/colab-badge\.svg\)]\()[\w:\/.-]+(?=\))")
nbviewer_link = re.compile(r"(?<=\[!\[Open nbviewer\]\(https:\/\/raw\.githubusercontent\.com\/pinecone-io\/examples\/master\/assets\/nbviewer-shield\.svg\)]\()[\w:\/.-]+(?=\))")


def link_valid(current_url: str, path: str, version: str) -> bool:
    # generate correct link
    if version == "colab":
        url = f"https://colab.research.google.com/github/pinecone-io/examples/blob/master/{path}"
    elif version == "nbviewer":
        url = f"https://nbviewer.org/github/pinecone-io/examples/blob/master/{path}"
    else:
        raise ValueError("version must be one of colab or nbviewer")
    # check if link is correct
    return current_url == url, url

def link_update(current_url: str, path: str, version: str) -> dict:
    # check link validity
    valid, url = link_valid(current_url, path, version)
    if not valid:
        # if link is not correct, update it
        with open(path, "r") as f:
            content = f.read()
        content = content.replace(current_url, url)
        with open(path, "w") as f:
            f.write(content)
    return {
        "updated": not valid,
        "past_url": current_url,
        "new_url": url
    }

def handle_no_shield(path: str, version: str, shield_error: bool) -> None:
    if shield_error:
        raise ValueError(f"No {version} shield found in {path}")
    else:
        logging.warning(f"No {version} shield found in {path}")


@click.group(help="Shields CLI")
def cli():
    pass


@click.command(help="Check if shields are up to date.")
@click.option("--update", default=False, help="Automatically update shield links.")
@click.option("--path", default=".", help="Path to check for shields.")
@click.option("--shield-error", default=False, help="Raise error if no shield is found.")
def run(update, path, shield_error):
    logging.basicConfig(level=logging.INFO)
    # get all notebook paths
    paths = [str(x) for x in Path(path).glob("**/*.ipynb")]
    logging.info(f"Found {len(paths)} notebooks")
    # check each notebook for shields
    for path in paths:
        with open(path, "r") as f:
            content = f.read()
        # try to find shields
        colab_url = colab_link.search(content)
        if colab_url:
            # if link exists, check it and update if incorrect
            colab_url = colab_url.group(0)
            if update:
                info = link_update(colab_url, path, "colab")
                if info["updated"]:
                    logging.info(f"Updated: {path}")
                else:
                    pass
            else:
                valid = link_valid(colab_url, path, "colab")
                if valid:
                    pass
                else:
                    logging.warning(f"Failed: {path}")
        else:
            handle_no_shield(path, "colab", shield_error)
        # now check nbviewer link
        nbviewer_url = nbviewer_link.search(content)
        if nbviewer_url:
            nbviewer_url = nbviewer_url.group(0)
            if update:
                info = link_update(nbviewer_url, path, "nbviewer")
                if info["updated"]:
                    logging.info(f"Updated: {path}")
                else:
                    pass
            else:
                valid = link_valid(nbviewer_url, path, "nbviewer")
                if valid:
                    pass
                else:
                    logging.warning(f"Failed: {path}")
        else:
            handle_no_shield(path, "nbviewer", shield_error)


cli.add_command(run)

if __name__ == "__main__":
    cli()
