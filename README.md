# Event classification using Transformers
This repository is forked from the "[official code developed for jet-tagging](https://github.com/jet-universe/particle_transformer)" using the so called "[Particle Transformer](https://arxiv.org/abs/2202.03772)".

Here, an adaptation for event classification is intended. Instead of classifying jets by extracting information from their constituents, the idea now is to classify proton-proton collisions simulated in the ATLAS detector by using the properties of their reconstructed objects (jets, leptons, missing energy transverse, etc.)

Since the framework also integrates other architectures (ParticleNet, PFN, P-CNN), different results and comparisons can be obtained directly using the same dataset as input. Since these architectures are quite optimal already, most of the changes come from the adaptation of the dataset that is given to the framework, since a particular format is required.

## Dataset
As a first approach, the "[DarkMachines dataset](https://arxiv.org/abs/2105.14027)" is used, which is available in csv format.
The procedure followed to convert this csv files into ROOT ntuples with the appropriate format is described in "[this repository](https://github.com/adrianrubio96/DarkMachines)".

## Plotting
Both the input variables and the metrics defined from the results are plotted using the "DarkMachines" branch of "[this repository](https://github.com/adrianrubio96/ROOTplotting/tree/DarkMachines)". Despite the instructions can be found there, the `DarkMachines` script already provides the needed shell scripts to plot the metrics: roc curve, confusion matrix, Loss and Accuracy functions, and scores distributions.


## Citations

If you use the Particle Transformer code and/or the JetClass dataset, please cite:

```
@InProceedings{Qu:2022mxj,
    author = "Qu, Huilin and Li, Congqiao and Qian, Sitian",
    title = "{Particle Transformer} for Jet Tagging",
    booktitle = "{Proceedings of the 39th International Conference on Machine Learning}",
    pages = "18281--18292",
    year = "2022",
    eprint = "2202.03772",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph"
}

@dataset{JetClass,
  author       = "Qu, Huilin and Li, Congqiao and Qian, Sitian",
  title        = "{JetClass}: A Large-Scale Dataset for Deep Learning in Jet Physics",
  month        = "jun",
  year         = "2022",
  publisher    = "Zenodo",
  version      = "1.0.0",
  doi          = "10.5281/zenodo.6619768",
  url          = "https://doi.org/10.5281/zenodo.6619768"
}
```

Additionally, if you use the ParticleNet model, please cite:

```
@article{Qu:2019gqs,
    author = "Qu, Huilin and Gouskos, Loukas",
    title = "{ParticleNet: Jet Tagging via Particle Clouds}",
    eprint = "1902.08570",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevD.101.056019",
    journal = "Phys. Rev. D",
    volume = "101",
    number = "5",
    pages = "056019",
    year = "2020"
}
```

For the QuarkGluon dataset, please cite:

```
@article{Komiske:2018cqr,
    author = "Komiske, Patrick T. and Metodiev, Eric M. and Thaler, Jesse",
    title = "{Energy Flow Networks: Deep Sets for Particle Jets}",
    eprint = "1810.05165",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "MIT-CTP 5064",
    doi = "10.1007/JHEP01(2019)121",
    journal = "JHEP",
    volume = "01",
    pages = "121",
    year = "2019"
}

@dataset{komiske_patrick_2019_3164691,
  author       = {Komiske, Patrick and
                  Metodiev, Eric and
                  Thaler, Jesse},
  title        = {Pythia8 Quark and Gluon Jets for Energy Flow},
  month        = may,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.3164691},
  url          = {https://doi.org/10.5281/zenodo.3164691}
}
```

For the TopLandscape dataset, please cite:

```
@article{Kasieczka:2019dbj,
    author = "Butter, Anja and others",
    editor = "Kasieczka, Gregor and Plehn, Tilman",
    title = "{The Machine Learning landscape of top taggers}",
    eprint = "1902.09914",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.21468/SciPostPhys.7.1.014",
    journal = "SciPost Phys.",
    volume = "7",
    pages = "014",
    year = "2019"
}

@dataset{kasieczka_gregor_2019_2603256,
  author       = {Kasieczka, Gregor and
                  Plehn, Tilman and
                  Thompson, Jennifer and
                  Russel, Michael},
  title        = {Top Quark Tagging Reference Dataset},
  month        = mar,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v0 (2018\_03\_27)},
  doi          = {10.5281/zenodo.2603256},
  url          = {https://doi.org/10.5281/zenodo.2603256}
}
```
