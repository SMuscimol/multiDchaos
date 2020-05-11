# multiDchaos
DOI: 10.5281/zenodo.3240447

Public code for the paper: 

Muscinelli, S.P., Gerstner, W. and Schwalger, T. Single neuron properties shape chaotic dynamics in random neural networks PLOS Computational Biology 15 (6), e1007122

Requirements
- Julia 0.7
- Julia libraries:
	- JLD
	- Optim

Usage:
The file to simulate the full rate network is "run_microscopic.jl", while the file to find the mean-field solution iteratively is "find-meanfield-solution-iteratively.jl". Such files relies on three custom julia modules: "RateNet", "AdaptiveNet" and "MyTools", included in the directories "ratenet", "adaptnet" and "mytools" respectively. Their respective paths should be included in your .juliarc in order for the code to work.
