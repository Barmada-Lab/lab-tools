# Survival Analysis

Runs survival analysis under a variety of conditions.

## Requirements:

- poetry

## Installation

1. Clone the parent repository, if you haven't already:
``` bash
$ git clone https://github.com/Barmada-Lab/lab-tools
```

2. install requirements:
``` bash
$ cd <lab_tools_location>/survival
$ poetry install
```

## Usage

To get a full description of the command line syntax and arguments, run

``` bash
$ poetry run surv --help
```

``` output 
usage: surv [-h] [--no-gedi] [--single-cell] [--save-stacks] [--save-masks] [--avg-reg] [--cpus CPUS] experiment_path scratch_path

positional arguments:
  experiment_path
  scratch_path

options:
  -h, --help       show this help message and exit
  --no-gedi        [NOT YET FUNCTIONAL] don't use rfp gedi
  --single-cell    [NOT YET FUNCTIONAL] use single cell tracking
  --save-stacks    save labelled stacks
  --save-masks     save labelled stacks
  --avg-reg        average registration across plate
  --cpus CPUS      num cpus
```

### Notes

- Experiments acquired using Lux should use the --avg-reg flag. Experiments acquired using the legacy acquisition script should _not_.
- Set the --cpus argument to the number of cpus available for fastest execution
- Note that scratch_path must already exist. You will have to create it manually.
- experiment_path should be a directory conforming to either the legacy experiment format or the new lux experiment format
- scratch_path may be any directory.

### Example

(Running on a greatlakes instance with 10 cpus)

``` bash
$ mkdir -p /scratch/sbarmada_root/sbarmada0/<your_uniqname>/.lab_tools/<experiment_name>

$ poetry run surv --cpus 10 --avg-reg /nfs/turbo/umms-sbarmada/experiments/<experiment_name> /scratch/sbarmada_root/sbarmada0/<your_uniqname>/.lab_tools/<experiment_name>
```