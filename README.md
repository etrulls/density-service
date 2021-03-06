# Remote U-Net service for Ilastik

This repository contains the a module wrapping pre-trained
[U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) models
for vesicle counting via density estimation, written in Pytorch. This component
can be called from Ilastik with the remote server plug-in, currently available
[here](https://github.com/etrulls/ilastik), which interfaces with the [remote
server](https://github.com/etrulls/cvlab-server), which in turns calls this
service. This work has been developed by the [Computer Vision lab at
EPFL](https://cvlab.epfl.ch) within the context of the Human Brain Project.

This repository is designed to be placed or symlinked inside the remote server
folder. A (possibly overtuned) list of requirements is given in `reqs.txt`, for
reference. This project corresponds to an as-of-now unpublished paper, so we
currently provide only pre-trained models and code for testing.

Pre-trained models are too large to be uploaded to Github: they are available
[here](http://icwww.epfl.ch/~trulls/shared/models_density.tar.gz) (just unzip
the tarball inside this folder). We provide three stacks to visualize results
(not used for training)
[here](http://icwww.epfl.ch/~trulls/shared/data.tar.gz).

![Teaser](https://raw.githubusercontent.com/etrulls/density-service/master/img/density_small.png "Teaser")
