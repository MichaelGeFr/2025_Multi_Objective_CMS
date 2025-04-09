# CMS 2025 conference in Enschede

## Usage

in moo_oven.py the multi-objective optimization code based on pymoo is available. In plotting the presented plots are available. 

### Installation
To install the project along with its development dependencies, execute the following command:

    poetry install --sync

Followed by

    poetry run pre-commit install

After this you are ready to perform the first commits to the repository.

Pre-commit ensures that the repository accepts your commit, automatically fixes some code styling problems and provides some hints for better coding.

### Adding dependencies
Adding dependencies to the project can be done via

    poetry add <package-name>@latest
