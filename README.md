# refineNCBF

Training neural conrol barrier functions and refining them with hamilton jacobi reachability.

## Installation

hj_reachability package requires the toofanian:hj_reachability fork, email alextoofanian@gmail.com for access.

## Usage

Example scripts and notebooks are in the `scripts` folder.

## Contributing

Repo is split into `data`, `scripts`, `tests`, and `refineNCBF` folders. All core functions are kept in the `refineNCBF` folder. Implementation scripts are
written in `scripts` folder. `data` folder contains all data files. `tests` folder contains all test scripts.

### refineNCBF folder structure

* training functions go in the training folder: `refineNCBF/training`
* refinement functions go in the refinement folder: `refineNCBF/refinement`
* shared specialized functions (ex: dynamic systems) go in the top level folder: `refineNCBF`
* basic util functions go in the utils folder: `refineNCBF/utils`