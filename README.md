# sculptCBF
"Selectively Carve Unsafe Local Perimeter Til CBF"

This repo is the bones behind my [UCSD masters thesis](https://escholarship.org/uc/item/4jc1v2sh).

(Note: The paper refers to "HJ Boundary Marching", which is equivalent to "sculptCBF")

### Abstract
Safe control policies for nonlinear dynamic systems are notoriously challenging to derive. Hamilton-Jacobi reachability analysis (HJ reachability) provides guaranteed safety in the form of the optimal control policy, but its compute cost scales exponentially with state space. Neural learning provides an alternative approach with its compute scaling only against problem complexity, but yields only approximate results. Recently, neural policies in the form of control barrier functions (CBFs) have been used to warmstart HJ reachability, yielding a guaranteed safe result more efficiently. However, a significant amount of compute is still spent to shape the CBF into the HJ reachability result. This paper introduces HJ Boundary Marching, which adapts the mechanics of warmstarted HJ reachability to refine erroneous control barrier function boundaries by minimally reshaping them to the nearest interior control-invariant boundary. This yields a guaranteed safe CBF for the same set as HJ reachability, with up to two orders of magnitude faster compute. A demonstration is provided on a 4-dimensional system with a learned neural CBF.

## Installation
Clone this repo, then run `pip install -e .` in the root directory.
additionally, this package the following forks:
* [hj_reachability](https://github.com/toofanian/hj_reachability) - Custom fork that accepts an "active" region for local dynamic programming, with no/minimal speedup.
* [optimized_dp](https://github.com/toofanian/optimized_dp) - Custom fork that accepts an "active" region for local dynamic programming, and has a speedup from only computing on subset of states.

Each of these packages provide different backend solvers for the sculptCBF algorithm, with differing performance. Note that `optimized_dp` is only available/tested on linux.

## Usage

Example scripts are in the `scripts` folder. The simplest reference is `scripts/local_hjr_solver/result_acc_march_jax.py`.

## Contributing

Repo is split into the intuitively named `data`, `scripts`, `tests`, and `refineNCBF` folders.

All core functions are kept in the `refineNCBF` folder. 

Implementation scripts are
written in `scripts` folder. 

Data files are kept in the `data` folder. 

Test are kept in `tests`.