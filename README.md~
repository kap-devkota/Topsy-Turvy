
# Table of Contents

1.  [Topsy-Turvy](#orgfb12c61)
    1.  [Summary](#orgad8f434)
    2.  [Files and Folders](#orgab55b6b)


<a id="orgfb12c61"></a>

# Topsy-Turvy


<a id="orgad8f434"></a>

## Summary

Topsy-Turvy is a novel method that synthesizes protein sequence information and 
inherent network structures transferrable across species to construct and/or enrich the PPI 
network for the target species.  

For more information about the model architectures (and downloading the pretained models
and datasets), go to [[]].


<a id="orgab55b6b"></a>

## Files and Folders

All the relevant test and evaluation codes are found inside the topsy<sub>turvy</sub> folder. 
The major files for training/evaluation are:

1.  embedding.py
2.  train.py
3.  evaluate.py

\`embedding.py\` is used to produce the PLM embeddings from the input sequence file in 
fasta format. It can be run using

    python embedding.py --seqs=<SEQ-FASTA-FILE> --o=<OUTPUT-DEST-FILE> --d=<GPU-DEVICE-ID>

\`train.py\` is used to train the model, given the sequence and network information for a source 
network. It can be run using

    python train.py [-h] 
          [--pos-pairs POS_PAIRS]               # Positive edgelist for training 
          [--neg-pairs NEG_PAIRS]               # Negative edgelist for training
          [--pos-test POS_TEST]                 # Positive edgelist for testing 
          [--neg-test NEG_TEST]                 # Negative edgelist for testing
          [--embedding EMBEDDING]               # PLM embedding obtained using `embedding.py`
          [--augment]                           # If (p, q) in training edgelist, add (q,p) for training too
          [--protein-size PROTEIN_SIZE]         # Maximum protein size to use in training data: default = 800
          [--projection-dim PROJECTION_DIM]     # Dimension of the projection layer: default 100
          [--dropout-p DROPOUT_P]               # Parameter p for the embedding dropout layer
          [--hidden-dim HIDDEN_DIM]             # Number of hidden units for comparison layer in contact prediction
          [--kernel-width KERNEL_WIDTH]         # The width of the conv. filter for contact prediction
          [--use-w]                             # Use the weight matrix in the interaction prediction or not
          [--do-pool]                           # 
          [--pool-width POOL_WIDTH] 
          [--sigmoid]
          [--negative-ratio NEGATIVE_RATIO] 
          [--epoch-scale EPOCH_SCALE]
          [--num-epochs NUM_EPOCHS] 
          [--batch-size BATCH_SIZE]
          [--weight-decay WEIGHT_DECAY] 
          [--lr LR] 
          [--lambda LAMBDA_]
          [--pred-skew PRED_SKEW] 
          [--skew-alpha SKEW_ALPHA] 
          [--use_glider]
          [--glider_param GLIDER_PARAM] 
          [--glider_thresh GLIDER_THRESH]
          [-o OUTPUT] 
          [--save-prefix SAVE_PREFIX] 
          [-d DEVICE]
          [--checkpoint CHECKPOINT] 

