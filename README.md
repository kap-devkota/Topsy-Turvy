
# Table of Contents

1.  [Topsy-Turvy](#orgfd11193)
    1.  [Summary](#org9484298)
    2.  [Files and Folders](#orgdce4c0d)
        1.  [Generating PLM Embeddings](#orgda00d19)
        2.  [Training the Model](#org528f09d)
        3.  [Evaluating Performance](#orgb61e327)
    3.  [Saved Models](#orge7fe3d1)



<a id="orgfd11193"></a>

# Topsy-Turvy


<a id="org9484298"></a>

## Summary

Topsy-Turvy is a novel method that synthesizes protein sequence information and 
inherent network structures transferrable across species to construct and/or enrich the PPI 
network for the target species.  

For more information about the model architectures (and downloading the pretained models
and datasets), go to [[]].


<a id="orgdce4c0d"></a>

## Files and Folders

All the relevant test and evaluation codes are found inside the \`topsy\_turvy\` folder. 
The major files for training/evaluation are:

1.  embedding.py
2.  train.py
3.  evaluate.py


<a id="orgda00d19"></a>

### Generating PLM Embeddings

\`embedding.py\` is used to produce the PLM embeddings from the input sequence file in 
fasta format. It can be run using

    python embedding.py 
          [--seqs SEQ-FASTA-FILE] 
          [--o OUTPUT-DEST-FILE]
          [--d GPU-DEVICE-ID]


<a id="org528f09d"></a>

### Training the Model

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
          [--do-pool]                           # Use the max pool layer
          [--pool-width POOL_WIDTH]             # The size of the max pool in the interaction model
          [--sigmoid]                           # Use sigmoid activation at the end of the interaction model: Default false
          [--negative-ratio NEGATIVE_RATIO]     # Number of negative training samples for each positive training sample
          [--epoch-scale EPOCH_SCALE]           # Report the heldout performance every multiple of this many epochs 
          [--num-epochs NUM_EPOCHS]             # Total number of epochs
          [--batch-size BATCH_SIZE]             # Minibatch size 
          [--weight-decay WEIGHT_DECAY]         # L2 regularization
          [--lr LR]                             # Learning rate
          [--lambda LAMBDA_]                    # The weight on the similarity objective
          # Use these parameter for Topsy-turvy training 
          [--use_glider]                        # Use this to train with Topsy-Turvy.
          [--glider_param GLIDER_PARAM]         # g_t param: default 0.2 
          [--glider_thresh GLIDER_THRESH]       # g_p param: Default 92.5
          # Output and device information
          [-o OUTPUT] 
          [--save-prefix SAVE_PREFIX] 
          [-d DEVICE]
          [--checkpoint CHECKPOINT] 

In order to use the \`train.py\` in Topsy-Turvy mode, add \`&#x2013;use\_glider\` option in the train.py.


<a id="orgb61e327"></a>

### Evaluating Performance

\`evaluate.py\` is used to perform evaluation, given the positive and negative test examples. It can be run using

    python evaluate.py
           [--model MODEL]                        # Model Location 
           [--pos-pairs POS_PAIRS]
           [--neg-pairs NEG_PAIRS]                # Positive and Negative Pairs
           [--embeddings EMBED]                   # PLM embedding file
           [--fasta FASTA]                        # If there is no PLM file, use FASTA to generate the PLM internally
           [--outfile OUTFILE]                    # Output prefix
           [--device DEVICE]                      # GPU device
           [--plot-prediction-distributions]
           [--plot-curves]


<a id="orge7fe3d1"></a>

## Saved Models

The best pre-trained models for D-SCRIPT and Topsy-Turvy are provided in the folder \`Pretrained-Models\` 

