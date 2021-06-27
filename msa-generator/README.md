# msa-generator

A main.ipynb file has been written and can be run for the initial downloading and extraction of required files

The essential steps are as follows:

- Extract scPDB.tar.gz into data/ folder
- Download uniref50.fasta and uniclust30_2017_10 into data/ folder
- Split the sequence.fasta file of every protein into respective chain.fasta file and remove the fasta files of DNA/RNA seqeuences
- The fasta files generated will have a lot of common sequences. Create a unique file that has only the common sequences, to speed up MSA generation
- To split the unique sequences file, run generate_splits.py to generate the splits to speed up processing of the MSAs on a SLURM cluster
- Modify pssm.sh SLURM script as needed to run the calculation of MSAs per split
- Eg. ```for i in {0..102} do; sbatch -o $i.out.%j -e $i.err.%j pssm.sh $i; done```
- for ```i in {0..102} do; python check.py $i; done``` can be used to check how many MSAs have been computed
