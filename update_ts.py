import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import argparse
import pathlib
from datetime import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("nb_path", help="path to notebooks")
args = parser.parse_args()

if __name__ == "__main__":
    file = pathlib.Path(args.nb_path)
    if file.is_file() and ".ipynb" in str(file):
        print(f"Processing... {file}")
        # read notebook
        with open(file) as f:
            nb = nbformat.read(f, as_version=4)

            # add code
            if "Latest update" in nb["cells"][0]["source"]:
                nb["cells"] = nb["cells"][1:]
            nb["cells"] = [
                nbformat.v4.new_markdown_cell(f"Latest update: {datetime.today()}")
            ] + nb["cells"]

            # execute notebook
            # ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            # ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

            # write notebook
            with open(file, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
