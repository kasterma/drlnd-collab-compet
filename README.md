# Project: Collaboration and Competition

Third project of the udacity nano-degree on reinforcement learning.

To setup run

    make setup
    
this will install the virtual env, and download e.g. the unity environment.

Then we can run the training by running

    python train.py train --run_id=<ID>
    
where <ID> is a numeric run identifier.  To then run the trained models without noise in the environment run

    python train.py evaluate --run_id=<ID>
    
# Environment

![Tennis environment](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)