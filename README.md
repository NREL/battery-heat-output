# Battery-Heat-Output-ML
Leveraging the 'Battery Failure Databank' published by NREL and NASA, uses machine-learning to predict fractional heat output during thermal runaway from cell metadata and measured ejected and remaining cell mass after a thermal runaway event. Fractional heat output, i.e., heat output from the cell body, from ejecta released out of the positive tab, and from ejecta released out of the negative tab, was measured using a fractional thermal runaway calorimeter. Cell metadata and ejected/remaining cell mass can be measured using substantially cheaper methods, enabling cheaper and faster estimation of thermal runaway risk.

## Repository structure
- 'data' folder contains a copy of the battery failure databank (version 2)
- 'src' folder contains classes and methods for data processing and model training
- `main.py` trains models and saves resulting predictions and errors into json files.

## Installation
Setup a virtual environment using conda with the 'environment.yml' file.