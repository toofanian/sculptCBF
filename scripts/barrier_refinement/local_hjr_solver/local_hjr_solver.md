# Local Hjr Solver
The local hjr solver algorithm loops over four stages:
1. active set pre-filter
   - the active set is pre-filtered to manually remove states that are not of interest
2. neighbor expansion
   - neighboring states are added to the active set 
3. value update
4. active set post-filter

## Classic
emulates the algorithm from somil & bajcsy. 

- algorithm:
  - expand active set to neighbors
  - compute vanilla hjr value update over active set 
  - remove unchanged values from active set 
  - repeat until active set is empty

## Only Decrease
emulates the algorithm from somil & bajcsy, but with modified value update which only 
accepts decreasing values. has the effect of only updating near the running zero 
levelset, since the zero levelset drives the values to decrease. the values far from
the viability kernel may be invalid because of this.

notable side effect: the viability kernel will be conservative (too small) 
if the initial zero levelset is not a complete superset of the viability kernel.
this is ok when finding the maximal control invariant subset of an initial set.

notable perk: forcing the values to only decrease gets rid of the local-oscillation
artifacts that occur in the classic algorithm, which prevented convergence detection.
    
- algorithm:
    - expand active set to neighbors
    - compute the only-decrease hjr value update over active set
    - remove unchanged values from active set
    - repeat until active set is empty
        
## ONLY ACTIVE NEAR ZERO LEVELSET
emulates the algorithm from somil & bajcsy, but with added active set filtering. the 
active set is filtered before each iteration to only include values near the zero
levelset. this is a like a hacky/forced version of only decrease, but has the benefit
of not requiring the initial zero levelset to be a complete superset of the viability
kernel. any guess will eventually lead to the correct viability kernel. the values far 
from the viability kernel may be invalid because of this.

-algorithm:
    - pre-filter active set to only include values near the zero levelset
    - expand active set to neighbors
    - compute vanilla hjr value update over active set
    - remove unchanged values from active set
    - repeat until active set is empty