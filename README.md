# WSU-NOvA-Vertexer
WSU efforts to vertex reconstruction in NOvA


--- 

_Original Author_: Michael Dolce mdolce@fnal.gov

Updated: Nov. 2023 

--- 

### Background
This repository is the home of the NOvA Neutrino Interaction Vertexing project at Wichita State University.

 **The objective of the NOvA vertexing is to improve the NOvA reconstruction algorithms of particle interactions within the detector to improve energy reconstruction**. This improved energy reconstruction, in turn, can help to improve a constraint on the neutrino interaction modeling and oscillation parameter constraints.

### CVN Vertexing 
This project uses a Convolutional Visual Network to identify the true vertex of NOvA neutrino interactions. **We use NOvA's pixel maps (or 'cvnmaps') as the features and the true vertex location as the labels**.

In this project, each vertex coordinate (x, y, z) is trained separately. So we have three models for each coordinate. The training for each coordinate is done with the `xz` and `yz` pixel maps of the detector.

The code is divided by each `{production, detector}` sample. More information can be found within each directory.

This project uses:
- `python 3.7.4`
- `tensorflow 2.3.1.` 


---
