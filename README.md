
# LSST Comet Interceptor Targer Reduction

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/lsst-comet-interceptor-target-reduction?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/lsst-comet-interceptor-target-reduction/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/parthenope-ap/lsst-comet-interceptor-target-reduction/smoke-test.yml)](https://github.com/parthenope-ap/lsst-comet-interceptor-target-reduction/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/parthenope-ap/lsst-comet-interceptor-target-reduction/branch/main/graph/badge.svg)](https://codecov.io/gh/parthenope-ap/lsst-comet-interceptor-target-reduction)
[![Read The Docs](https://img.shields.io/readthedocs/lsst-comet-interceptor-target-reduction)](https://lsst-comet-interceptor-target-reduction.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/parthenope-ap/lsst-comet-interceptor-target-reduction/asv-main.yml?label=benchmarks)](https://parthenope-ap.github.io/lsst-comet-interceptor-target-reduction/)

## Installation

```bash
git clone https://github.com/parthenope-ap/lsst-comet-interceptor-target-reduction.git
cd lsst-comet-interceptor-target-reduction
conda env create -f environment.yml
conda activate lsst-comet
```

## Examples

### Running the Scripts
```bash
python main.py --help 
```
or
```bash
python parallelized_main.py --help
```

Make sure to activate the Conda environment before running these scripts:
```bash
conda activate lsst-comet
```

