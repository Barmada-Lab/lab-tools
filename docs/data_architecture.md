# Barmada Lab Data Architecture

## Summary

Our lab produces a lot of microscopy data. Our challenge is to organize that data according to utilization while ensuring we don't lose anything. We also want to minimize the cost and complexity of our solution.

## Definitions

The base unit of organization for our data is the __experiment__. Experiments are collections of microscopy images and associated analyses. They have globally unique, descriptive names, and exist on disk either as directories or tarballs.

We organize our experiments into collections called __datastores__ based on availability requirements. 

__acquisition__ datastores persist data while acquisitions are ongoing.

Experiments move to the __analysis__ datastore for analysis. The analysis datastore is embodied by the "Turbo" fileshare.

The __archive__ datastore holds data for long-term storage until is is no longer needed. The archive is embodied by "dataden." 

## Requirements

In an ideal world, we would only need one datastore; all processes related to data generation and analysis would operate on that central datastore, and we would have no need for data architecture. In reality, technical constraints require that we partition our storage depending on availability requirements and storage capacity. 

which is a tape-backed archive having the largest capacity and lowest availability of all the datastores. 

Availability requirements change depending on the age of an experiment. 

During aquisition, experiments need to live close to the instrument performing the acquistion. 
## Architecture

## Tech

## Configuring a new node
