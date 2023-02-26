# PAM_P3S

## Authors
------------ 

[Pierre Chouteau](mailto:pierre.chouteau@atiam.fr), [Henri Desvallées](mailto:henri.desvallees@atiam.fr), [Louis Lalay](mailto:louis.lalay@atiam.fr), [Mathilde Lefebvre](mailto:mathilde.lefebvre@atiam.fr), [Ivan Meresman-Higgs](mailto:meresmanhiggs@atiam.fr), [Théo Nguyen](mailto:theo.nguyen@atiam.fr)


## Introduction
------------ 

This project is part of the [ATIAM master](https://www.atiam.ircam.fr/en/). The aim of this project is to separate instruments from a quintet as if each instrument was recorded alone, but they indeed have been recorded playing together. This technique is called source separation. A effort was made to link signal processing with acoustics and sound recording in order to make better informed choices, mainly when choosing and placing the microphones. 

You can read our report [SourceSeparation](./report/PAM_SourceSeparation.pdf) and  hear some of our results on our [website](https://sourceseparation2022-2023.github.io/).


## Abstract
------------ 

Music Source Separation (MSS) is the process of separating individual audio signals from a mixed recording containing multiple sound sources, such as different musical instruments, vocals and ambient noise, and its various applications include remixing, transcription and music recommendation. 
In the context of real acoustic recordings, the separation task is particularly challenging due to the complexity and variability of acoustic instruments and recording conditions such as room acoustics and microphone directivity.%,  add some challenges for a non-degraded MSS. 
We propose the use of Non-Negative Matrix Factorization (NMF) separation algorithms, where an easily interpretable time-frequency representation of the power spectrogram aims to decompose each instrument recording into a dictionary of notes (frequency basis) and a time activation basis (time basis), reminiscent of a musical score. 
In our multi-channel setting, we aim to implement efficient, conditioned versions of this algorithm to be applied to musical recordings performed in a known and controlled context, to investigate methods of informing this algorithm for improved performance.
To this end, we conducted a professional-level recording of a chamber music quintet.
We implemented a set of standard NMF algorithms that can be conditioned on temporal and spectral information from the instruments that were specifically registered at the time of the recording for this purpose.
    

*Keywords*: Music, Source separation, Non-Negative Matrix Factorization, Live Recording, Acoustics

## Install
------------ 

To run our scripts, you will need to have a specific environment which will require the installation of miniconda (or anaconda). 
If you do not already have it, you can install it from the original [website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).


- Clone the github repository

``` 
git clone https://github.com/PierreChouteau/PAM_P3S.git
``` 

- Create a virtual env with python 3.10.8:

``` 
conda create -n P3S python=3.10.8
``` 

- Activate the environment:
``` 
conda activate P3S
``` 

- Go into the repository and install the dependencies: 
``` 
cd PAM_P3S
pip install -r requirement.txt
``` 

- Install the src package:
```
pip install -e .
```