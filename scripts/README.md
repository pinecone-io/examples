# Examples Admin Scripts

This directory contains helper scripts for admin of the /examples repo.

## Shields Checker

The beginning of every notebook in the examples repo should include a Colab and nbviewer shield to allow easy navigation to either service. To confirm validity and update changing links we can use the **Shields Checker** script.

To use the script navigate to the root directory of the examples repo and run the following in a terminal window:

```
python scripts/shields-checker.py run --path . --shield-error False --update False
```

This will run the shields checker script across all notebooks in the directory, it will not update shield links, and it will not raise an error if no shields are found in a notebook.

We can adjust the default parameters depending on our intended use:

* `--path` allows us to specify a specific directory like `learn` or `docs`. Default value is `.` (search all directories).
* `--shield-error` allows us to raise a `ValueError` if set to `True` and if no shields are found in a notebook. Default value is `False` which logs a warning to the console but does not raise an error.
* `--update` specifies whether shield links should be automatically updated. When set to `True` both Colab and nbviewer links will be updated *if* they are found to be invalid. Default value is `False` which only logs whether links are valid or not.
